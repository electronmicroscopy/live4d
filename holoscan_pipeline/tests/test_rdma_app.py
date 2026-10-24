"""Application-level acquisition rotation against a local imagemux protocol peer."""
import queue
import socket
import threading
from types import SimpleNamespace

import pytest

from test_rdma_batches import LOCAL_ADDR, Loopback, RingWriter, _start, _image, _end, pytestmark


@pytest.mark.parametrize("devices", [(0,), (0, 0), (0, 1)])
def test_application_routes_complete_acquisitions_and_reuses_branches(devices):
    import cupy as cp
    import numpy as np
    from dectris import imxrdma
    from dectris.imagemux import (HELLO_LEN, READY_LEN, Accept, Hello,
                                  QpEndpoint, _read_exact)
    from live4d.app import PipelineApp
    from live4d.config import PipelineConfig

    if cp.cuda.runtime.getDeviceCount() <= max(devices):
        pytest.skip("requires two visible GPUs")
    acquisitions = [((96, 96), 0), ((192, 192), 3), ((192, 192), 5), ((96, 96), 1)]
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(20)
    claims, permits, errors = queue.Queue(), queue.Queue(), []
    observed, results, registrations = [], [], {}
    stopping = threading.Event()

    def sender():
        context = imxrdma.Context(local_addr=LOCAL_ADDR)
        try:
            for index, (shape, count) in enumerate(acquisitions):
                sock, _ = listener.accept()
                peer = Loopback.__new__(Loopback)
                peer.ctx = context
                peer.tx = peer.source = peer.source_mr = None
                peer.credit = np.zeros(1, dtype=np.uint64)
                peer.credit_mr = context.register(peer.credit.ctypes.data, 8, remote_write=True)
                try:
                    with sock:
                        sock.settimeout(20)
                        hello = Hello.decode(_read_exact(sock, HELLO_LEN))
                        peer.ring_bytes = hello.ring_len
                        peer.ring_mr = SimpleNamespace(address=hello.ring_addr, rkey=hello.ring_rkey)
                        peer.imms = np.empty(64, dtype=np.uint32)
                        peer.tx = context.create_qp(recv_depth=1, send_depth=64, signal_every=1)
                        e = hello.endpoint
                        peer.tx.connect(imxrdma.Endpoint(e.qp_num, e.psn, e.lid, e.mtu, e.gid))
                        e = peer.tx.endpoint
                        sock.sendall(Accept(QpEndpoint(e.qp_num, e.psn, e.lid, e.mtu, e.gid),
                                            peer.credit_mr.address, peer.credit_mr.rkey).encode())
                        assert len(_read_exact(sock, READY_LEN)) == READY_LEN
                        claim = permits.get(timeout=20)
                        assert claim["acquisition"] == index
                        device = devices[index % len(devices)]
                        assert claim["device"] == device
                        identity = (hello.ring_addr, hello.ring_rkey)
                        lane = index % len(devices)
                        assert registrations.setdefault(lane, identity) == identity
                        writer = RingWriter(peer.ring_bytes)
                        _start(writer, 501 + index, count, shape)
                        if count:
                            _image(writer, 501 + index, list(range(count)),
                                   [index + 7] * count, shape)
                        _end(writer, 501 + index)
                        peer.send(writer.records)
                        assert sock.recv(1) == b""  # Receiver closes only after native detach.
                finally:
                    for resource in (peer.tx, peer.source_mr, peer.credit_mr):
                        if resource is not None:
                            resource.close()
        except BaseException as error:
            errors.append(error)
        finally:
            listener.close()

    def observer(batch, positioned):
        if len(batch.image_ids):
            observed.append((int(batch.series_id), cp.cuda.runtime.getDevice(),
                             list(batch.image_ids), cp.asnumpy(positioned.images)))
        if batch.end_of_series:
            results.append(dict(positioned.result))
            if len(results) == len(acquisitions):
                stopping.set()

    config = PipelineConfig(transport="rdma", devices=devices, batch_size=2, queue_depth=2,
        execute=dict(mode="service"), scan=dict(width=4, height=2),
        rdma=dict(remote_address=f"127.0.0.1:{listener.getsockname()[1]}",
                  local_address=LOCAL_ADDR, ring_memory="host_mapped", ring_mib=1))
    app = PipelineApp(config, observer=observer, receiver_ready=claims.put)
    assert app.plan.ok, app.plan.summary()
    for resources in (app.branch_ready, app.windows, app.rdma.rings,
                      app.rdma.sessions, app.rdma.results):
        assert len(resources) == len({id(item) for item in resources}) == len(devices)
    producer = threading.Thread(target=sender)
    producer.start()

    def run():
        try:
            app.run(stop_event=stopping)
        except BaseException as error:
            errors.append(error)
    consumer = threading.Thread(target=run)
    consumer.start()
    try:
        for index in range(len(acquisitions)):
            claim = claims.get(timeout=20)
            assert claim["acquisition"] == index
            permits.put(claim)
        consumer.join(20)
        producer.join(20)
        assert not consumer.is_alive() and not producer.is_alive()
        assert not errors, errors
        assert len(results) == len(acquisitions)
        for index, (shape, count) in enumerate(acquisitions):
            sid, device, value = 501 + index, devices[index % len(devices)], index + 7
            result, = [r for r in results if r["series_id"] == sid]
            assert result["device"] == device
            assert result["images"] == result["announced_images"] == result["positions"] == count
            assert result["gap_events"] == result["missing_images"] == 0
            y, x = np.ogrid[:shape[0], :shape[1]]
            squared = (2*y-(shape[0]-1))**2 + (2*x-(shape[1]-1))**2
            bf = int((squared <= 4*16**2).sum())
            adf = int(((squared >= 4*24**2) & (squared <= 4*48**2)).sum())
            assert result["bf_sum"] == bf * value * count
            assert result["adf_sum"] == adf * value * count
            batches = [(d, ids, pixels) for s, d, ids, pixels in observed if s == sid]
            assert [i for _, ids, _ in batches for i in ids] == list(range(count))
            for d, ids, pixels in batches:
                assert d == device and tuple(pixels.shape) == (len(ids), *shape)
                np.testing.assert_array_equal(pixels, np.full(pixels.shape, value, np.uint16))
        assert len(set(registrations.values())) == len(devices)
        with pytest.raises(RuntimeError, match="new RDMA PipelineApp"):
            app.run()
    finally:
        stopping.set()
        consumer.join(20)
        producer.join(20)
