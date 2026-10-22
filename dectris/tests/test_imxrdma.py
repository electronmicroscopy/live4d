"""imxrdma on a real RDMA port: two queue pairs connected to each other play imagemux and the
receiver. Needs the module built (native/) and IMXRDMA_LOCAL_ADDR set to the address of an
active RoCE interface, e.g.

    IMXRDMA_LOCAL_ADDR=10.42.41.11 PYTHONPATH=build python3 -m unittest tests.test_imxrdma
"""

import ctypes
import os
import sys
import time
import unittest

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from dectris.imagemux import FrameHeader, FrameKind, FrameReader, frame_span  # noqa: E402

try:
    from dectris import imxrdma
except ImportError:
    imxrdma = None

LOCAL_ADDR = os.environ.get("IMXRDMA_LOCAL_ADDR")
RTS, ERR = 3, 6


def _cudart():
    for name in ("libcudart.so.13", "libcudart.so.12", "libcudart.so"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            pass
    return None


@unittest.skipUnless(imxrdma and LOCAL_ADDR, "needs imxrdma and IMXRDMA_LOCAL_ADDR")
class LoopbackTest(unittest.TestCase):
    RING = 1 << 16

    def setUp(self):
        self.ctx = imxrdma.Context(local_addr=LOCAL_ADDR)
        self.ring = np.zeros(self.RING, dtype=np.uint8)
        self.src = np.zeros(self.RING, dtype=np.uint8)
        self.credit = np.zeros(1, dtype=np.uint64)
        self.ring_mr = self.ctx.register(self.ring.ctypes.data, self.ring.nbytes, remote_write=True)
        self.src_mr = self.ctx.register(self.src.ctypes.data, self.src.nbytes)
        self.credit_mr = self.ctx.register(self.credit.ctypes.data, 8, remote_write=True)
        self.rx = self.ctx.create_qp(recv_depth=64, send_depth=8, signal_every=4)
        self.tx = self.ctx.create_qp(recv_depth=1, send_depth=64)
        self.rx.connect(self.tx.endpoint)
        self.tx.connect(self.rx.endpoint)
        self.imms = np.empty(64, dtype=np.uint32)

    def tearDown(self):
        for o in (self.rx, self.tx, self.ring_mr, self.src_mr, self.credit_mr):
            o.close()

    def poll(self, qp, want=1, timeout=5.0):
        got, deadline = [], time.monotonic() + timeout
        while len(got) < want and time.monotonic() < deadline:
            n, _, failed = qp.poll(self.imms)
            if failed:
                return got, failed
            got += self.imms[:n].tolist()
        return got, 0

    def test_context(self):
        self.assertTrue(self.ctx.device)
        self.assertEqual(self.ctx.gid[10:12], b"\xff\xff")  # IPv4-mapped RoCE v2 GID
        self.assertEqual(self.rx.endpoint.gid, self.ctx.gid)
        self.assertEqual((self.rx.state, self.tx.state), (RTS, RTS))

    def test_unknown_address(self):
        with self.assertRaisesRegex(RuntimeError, "no RDMA port"):
            imxrdma.Context(local_addr="192.0.2.1")

    def test_frames_through_the_reader(self):
        """What imagemux does: header + data into the ring, then WRITE_WITH_IMM with the
        sequence number; the reader finds the frames, the credit goes back."""
        self.assertEqual(self.rx.post_recvs(100), 64)  # capped at recv_depth
        position = 0
        for seq in range(3):
            data = bytes([seq + 1]) * 1000
            header = FrameHeader(FrameKind.IMAGE, seq, position, len(data), series_id=7).encode()
            frame = header + data
            self.src[position:position + len(frame)] = np.frombuffer(frame, dtype=np.uint8)
            self.assertTrue(self.tx.write(self.src_mr, position, len(frame),
                                          self.ring_mr.address + position, self.ring_mr.rkey,
                                          imm=seq))
            position += frame_span(len(data))
        imms, failed = self.poll(self.rx, want=3)
        self.assertEqual((imms, failed), ([0, 1, 2], 0))
        reader = FrameReader(self.RING)
        for seq, imm in enumerate(imms):
            frame = reader.next(self.ring, imm)
            self.assertEqual((frame.header.sequence, frame.header.series_id), (seq, 7))
            self.assertEqual(bytes(self.ring[frame.data_offset:frame.data_offset + 3]),
                             bytes([seq + 1]) * 3)
        self.assertEqual(self.rx.post_recvs(3), 3)
        self.assertEqual(self.rx.stats["recvs_outstanding"], 64)

        self.assertTrue(self.rx.write_u64(reader.position, self.credit_mr.address,
                                          self.credit_mr.rkey))
        deadline = time.monotonic() + 5
        while int(self.credit[0]) != reader.position and time.monotonic() < deadline:
            self.rx.poll(self.imms)
        self.assertEqual(int(self.credit[0]), reader.position)

    def test_send_queue_full_then_freed(self):
        posted = 0
        while self.rx.write_u64(posted, self.credit_mr.address, self.credit_mr.rkey):
            posted += 1
        self.assertEqual(posted, 8)  # send_depth
        deadline = time.monotonic() + 5
        while self.rx.stats["writes_outstanding"] and time.monotonic() < deadline:
            self.rx.poll(self.imms)
        self.assertEqual(self.rx.stats["writes_outstanding"], 0)
        self.assertTrue(self.rx.write_u64(1, self.credit_mr.address, self.credit_mr.rkey))

    def test_bad_rkey_is_reported(self):
        self.rx.post_recvs(4)
        self.assertTrue(self.tx.write(self.src_mr, 0, 128, self.ring_mr.address,
                                      self.ring_mr.rkey ^ 0xFFFF, imm=0))
        _, failed = self.poll(self.tx, want=1)
        self.assertGreater(failed, 0)
        self.assertIn("remote access error", self.tx.errors)
        self.assertEqual(self.tx.state, ERR)

    @unittest.skipUnless(_cudart(), "needs libcudart")
    def test_gpu_ring(self):
        """A ring in GPU memory, registered like host memory (nvidia_peermem)."""
        cuda = _cudart()
        ptr, n = ctypes.c_void_p(), 1 << 20
        self.assertEqual(cuda.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(n)), 0)
        try:
            mr = self.ctx.register(ptr.value, n, remote_write=True)
            self.rx.post_recvs(1)
            self.src[:256] = np.arange(256, dtype=np.uint8)
            self.assertTrue(self.tx.write(self.src_mr, 0, 256, mr.address, mr.rkey, imm=42))
            self.assertEqual(self.poll(self.rx), ([42], 0))
            out = np.zeros(256, dtype=np.uint8)
            self.assertEqual(cuda.cudaMemcpy(ctypes.c_void_p(out.ctypes.data), ptr,
                                             ctypes.c_size_t(256), 2), 0)  # device to host
            np.testing.assert_array_equal(out, self.src[:256])
            mr.close()
        finally:
            cuda.cudaFree(ptr)


if __name__ == "__main__":
    unittest.main()
