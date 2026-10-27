"""Wire format, ring reader and handshake of the imagemux transport (no RDMA, no holoscan)."""

import os
import socket
import sys
import threading
import unittest

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from dectris.imagemux import (  # noqa: E402
    ACCEPT_LEN, FRAME_HEADER_LEN, HELLO_LEN, READY_LEN, PROTOCOL_VERSION, Accept, Compression,
    CreditTracker, DataType, QpEndpoint, FrameHeader, FrameKind, FrameReader, Hello, ImageHeader,
    ImagemuxCodec, LocalInfo, RemoteRegion, TcpConnector, HandshakeError, _read_exact,
    decode_ready, encode_ready, frame_span, padded)

ENDPOINT = QpEndpoint(0x1234, 0xABCDEF, 7, 5, bytes([0] * 10 + [0xFF, 0xFF, 10, 0, 0, 1]))


class RingWriter:
    """Writes frames into a bytearray ring the way imagemux does."""

    def __init__(self, ring_len):
        self.ring = bytearray(ring_len)
        self.position = 0
        self.sequence = 0

    def write(self, kind, data=b"", **kw):
        offset = self.position % len(self.ring)
        span = frame_span(len(data))
        if kind != FrameKind.WRAP and offset + span > len(self.ring):
            self.write(FrameKind.WRAP)
            offset = 0
        header = FrameHeader(kind, self.sequence, self.position, len(data), **kw)
        self.ring[offset:offset + FRAME_HEADER_LEN] = header.encode()
        self.ring[offset + FRAME_HEADER_LEN:offset + FRAME_HEADER_LEN + len(data)] = data
        self.position += len(self.ring) - offset if kind == FrameKind.WRAP else span
        self.sequence += 1
        return self.sequence - 1


class WireTest(unittest.TestCase):
    def test_handshake_roundtrip(self):
        hello = Hello(ENDPOINT, 0x7F00_0000_0000, 1 << 30, 0xDEADBEEF)
        buf = hello.encode()
        self.assertEqual((len(buf), buf[:4]), (HELLO_LEN, b"IMXR"))
        self.assertEqual(Hello.decode(buf), hello)
        accept = Accept(ENDPOINT, 0x5555_0000, 42)
        self.assertEqual(len(accept.encode()), ACCEPT_LEN)
        self.assertEqual(Accept.decode(accept.encode()), accept)

    def test_hello_rejects_misaligned_ring(self):
        for addr, length in [(0x1000, 4096 + 32), (0x1000, 128), (0x1040, 4096)]:
            with self.assertRaises(HandshakeError):
                Hello.decode(Hello(ENDPOINT, addr, length, 1).encode())

    def test_ready_rejects_bad_version(self):
        buf = encode_ready()
        self.assertEqual(len(buf), READY_LEN)
        decode_ready(buf)
        with self.assertRaises(HandshakeError):
            decode_ready(buf[:4] + b"\x01\x00" + buf[6:])

    def test_frame_header_layout(self):
        image = ImageHeader(0x0505, 0x0606, 0x0707, 0x0808, 0x0909, DataType.UINT16,
                            Compression.BSLZ4, 0x0A0A, 0x0B0B, 0x0C0C)
        buf = FrameHeader(FrameKind.IMAGE, 0x0101, 0x0202, 0x0303, 0x0404, image).encode()
        u = lambda o, n: int.from_bytes(buf[o:o + n], "little")  # noqa: E731
        self.assertEqual([u(4, 2), u(6, 2), u(8, 8), u(16, 8), u(24, 8), u(32, 8)],
                         [2, 2, 0x0101, 0x0202, 0x0303, 0x0404])
        self.assertEqual([u(40, 8), u(48, 8), u(56, 4), u(60, 4), u(64, 4), u(68, 2), u(70, 2),
                          u(72, 8), u(80, 2), u(82, 2)],
                         [0x0505, 0x0606, 0x0707, 0x0808, 0x0909, 2, 1, 0x0A0A, 0x0B0B, 0x0C0C])
        self.assertEqual((len(buf), set(buf[84:])), (FRAME_HEADER_LEN, {0}))
        self.assertEqual(FrameHeader.decode(buf).image, image)

    def test_frame_header_rejects_unknown_values(self):
        buf = bytearray(FrameHeader(FrameKind.IMAGE, 0, 0, 0).encode())
        for offset in (6, 68, 70):
            bad = bytearray(buf)
            bad[offset] = 9
            with self.assertRaises(HandshakeError):
                FrameHeader.decode(bytes(bad))

    def test_spans(self):
        self.assertEqual([padded(n) for n in (0, 1, 128, 129)], [0, 128, 128, 256])
        self.assertEqual([frame_span(n) for n in (0, 1, 128, 129)], [128, 256, 256, 384])


class ReaderTest(unittest.TestCase):
    def test_frames_in_order_with_wrap(self):
        w = RingWriter(1024)
        reader = FrameReader(1024)
        seen = []
        for i in range(10):
            data = bytes([i]) * 300  # span 512: the third frame does not fit and wraps
            imm = w.write(FrameKind.IMAGE, data)
            # A wrap marker is announced by its own immediate.
            while reader.sequence <= imm:
                frame = reader.next(w.ring, reader.sequence & 0xFFFFFFFF)
                if frame:
                    start = frame.data_offset
                    seen.append(bytes(w.ring[start:start + frame.header.data_size]))
        self.assertEqual(seen, [bytes([i]) * 300 for i in range(10)])
        self.assertEqual(reader.position, w.position)

    def test_out_of_order_immediate(self):
        w = RingWriter(1024)
        w.write(FrameKind.START, b"x")
        with self.assertRaises(HandshakeError):
            FrameReader(1024).next(w.ring, 1)

    def test_error_shows_the_bytes_read(self):
        with self.assertRaisesRegex(HandshakeError, r"bad magic.*ring offset 0.*first bytes 00"):
            FrameReader(1024).next(bytearray(1024), 0)

    def test_header_that_shows_up_late_is_waited_for(self):
        w = RingWriter(1024)
        w.write(FrameKind.START, b"x")

        class LateRing:  # the NIC's write into GPU memory becomes visible on the third read
            reads = 0

            def __getitem__(self, key):
                LateRing.reads += 1
                return w.ring[key] if LateRing.reads >= 3 else bytes(FRAME_HEADER_LEN)

        frame = FrameReader(1024, settle_s=1.0).next(LateRing(), 0)
        self.assertEqual((frame.header.kind, LateRing.reads), (FrameKind.START, 3))
        LateRing.reads = -10**6  # never shows up: give up after the settle time
        with self.assertRaises(HandshakeError):
            FrameReader(1024, settle_s=0.01).next(LateRing(), 0)

    def test_stale_header(self):
        w = RingWriter(1024)
        with self.assertRaises(HandshakeError):  # nothing written at the position yet
            FrameReader(1024).next(w.ring, 0)


class FakeEvent:
    def __init__(self, done=False):
        self.done = done


class CreditTrackerTest(unittest.TestCase):
    def test_nothing_pending(self):
        self.assertEqual(CreditTracker().poll(), 0)

    def test_work_without_events_is_credited_at_once(self):
        credits = CreditTracker()
        credits.add(300)
        self.assertEqual(credits.poll(), 300)

    def test_credited_in_order_once_the_gpu_is_done(self):
        credits = CreditTracker()
        first, second = FakeEvent(), FakeEvent()
        credits.add(100, [first])
        credits.add(250, [second])
        second.done = True  # a later batch finished first: still nothing may be credited
        self.assertEqual(credits.poll(), 0)
        first.done = True
        self.assertEqual(credits.poll(), 250)

    def test_all_events_of_a_batch_are_needed(self):
        credits = CreditTracker()
        a, b = FakeEvent(True), FakeEvent()
        credits.add(100, [a, b])
        self.assertEqual(credits.poll(), 0)
        b.done = True
        self.assertEqual(credits.poll(), 100)

    def test_position_never_goes_back(self):
        credits = CreditTracker()
        credits.add(100, [FakeEvent(True)])
        self.assertEqual(credits.poll(), 100)
        self.assertEqual(credits.poll(), 100)


class MockImagemux:
    """Listens like imagemux's handshake port and plays its side of the handshake."""

    def __init__(self, hang_up_after_handshake=False):
        self.srv = socket.socket()
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(1)
        self.port = self.srv.getsockname()[1]
        self.hello = None
        self.ready = False
        self.hang_up = hang_up_after_handshake
        self.receiver_closed = threading.Event()
        self.thread = threading.Thread(target=self.serve, daemon=True)
        self.thread.start()

    def serve(self):
        conn, _ = self.srv.accept()
        with conn, self.srv:
            self.hello = Hello.decode(_read_exact(conn, HELLO_LEN))
            conn.sendall(Accept(ENDPOINT, 0x2000, 0xCD).encode())
            try:
                decode_ready(_read_exact(conn, READY_LEN))
            except ConnectionError:
                return  # the receiver gave up before Ready (e.g. attaching the QP failed)
            self.ready = True
            if not self.hang_up:
                conn.recv(1)  # stay connected until the receiver closes
                self.receiver_closed.set()


class HandshakeTest(unittest.TestCase):
    RING = RemoteRegion(0x1000, 1 << 20, 0xAB)

    def connector(self, imagemux, attach=lambda peer: 3):
        return TcpConnector("127.0.0.1", ImagemuxCodec(), lambda: LocalInfo(self.RING, ENDPOINT),
                            attach, port=imagemux.port)

    def test_handshake_exchanges_regions_and_endpoints(self):
        imagemux = MockImagemux()
        attached = []
        link = self.connector(imagemux, lambda peer: attached.append(peer) or 3).connect()
        imagemux.thread.join(0.5)
        self.assertEqual(imagemux.hello, Hello(ENDPOINT, 0x1000, 1 << 20, 0xAB))
        self.assertTrue(imagemux.ready)
        self.assertEqual(attached[0].endpoint, ENDPOINT)
        self.assertEqual((link.conn, link.peer.credit), (3, RemoteRegion(0x2000, 8, 0xCD)))
        self.assertTrue(link.liveness.alive)
        link.close()

    def test_ready_is_sent_after_the_qp_is_attached(self):
        imagemux = MockImagemux()
        link = self.connector(imagemux, lambda peer: self.assertFalse(imagemux.ready) or 3).connect()
        imagemux.thread.join(0.5)
        self.assertTrue(imagemux.ready)
        link.close()

    def test_imagemux_hangup_marks_dead(self):
        link = self.connector(MockImagemux(hang_up_after_handshake=True)).connect()
        self.assertTrue(link.liveness.wait_dead(2))
        self.assertFalse(link.liveness.alive)
        link.close()

    def test_local_close_wakes_liveness_and_is_idempotent(self):
        imagemux = MockImagemux()
        link = self.connector(imagemux).connect()
        imagemux.thread.join(0.5)
        link.close()
        self.assertTrue(link.liveness.wait_dead(2))
        link.close()

    def test_local_close_hangs_up_on_imagemux_at_once(self):
        imagemux = MockImagemux()
        link = self.connector(imagemux).connect()
        imagemux.thread.join(0.5)
        link.close()
        self.assertTrue(imagemux.receiver_closed.wait(2))

    def test_imagemux_closing_after_the_hello_is_explained(self):
        server = socket.socket()
        server.bind(("127.0.0.1", 0))
        server.listen(1)

        def serve():
            conn, _ = server.accept()
            with conn, server:
                _read_exact(conn, HELLO_LEN)  # reads the Hello, then hangs up without an Accept

        threading.Thread(target=serve, daemon=True).start()
        connector = TcpConnector("127.0.0.1", ImagemuxCodec(), lambda: LocalInfo(self.RING, ENDPOINT),
                                 lambda peer: 3, port=server.getsockname()[1])
        with self.assertRaisesRegex(ConnectionError, "after our Hello.*imagemux's log"):
            connector.connect()

    def test_attach_failure_closes_socket(self):
        imagemux = MockImagemux()

        def boom(peer):
            raise RuntimeError("no queue pair")

        with self.assertRaises(RuntimeError):
            self.connector(imagemux, boom).connect()
        imagemux.thread.join(2)
        self.assertFalse(imagemux.ready)


class ProbeTest(unittest.TestCase):
    """tools/probe_imagemux.py against a fake imagemux that only speaks one protocol version."""

    def fake_imagemux(self, speaks):
        server = socket.socket()
        server.bind(("127.0.0.1", 0))
        server.listen(5)

        def serve():
            with server:
                for _ in range(8):
                    try:
                        conn, _ = server.accept()
                    except OSError:
                        return
                    with conn:
                        try:
                            data = _read_exact(conn, HELLO_LEN)
                        except ConnectionError:
                            continue  # the client hung up before sending a whole Hello
                        if int.from_bytes(data[4:6], "little") == speaks:
                            conn.sendall(Accept(ENDPOINT, 0x2000, 0xCD).encode())

        threading.Thread(target=serve, daemon=True).start()
        return server.getsockname()[1]

    def test_hello_carries_the_version_and_is_otherwise_valid(self):
        from tools.probe_imagemux import hello_bytes
        data = hello_bytes("10.42.41.11", PROTOCOL_VERSION)
        hello = Hello.decode(data)
        self.assertEqual(hello.endpoint.gid[10:], b"\xff\xff\x0a\x2a\x29\x0b")
        patched = hello_bytes("10.42.41.11", 7)
        self.assertEqual(int.from_bytes(patched[4:6], "little"), 7)
        self.assertEqual(patched[:4] + patched[6:], data[:4] + data[6:])
        with self.assertRaises(HandshakeError):
            Hello.decode(patched)

    def test_tells_the_versions_apart(self):
        from tools.probe_imagemux import probe
        port = self.fake_imagemux(speaks=PROTOCOL_VERSION + 1)
        self.assertIn("closed by imagemux", probe("127.0.0.1", port, "10.42.41.11", PROTOCOL_VERSION))
        self.assertIn("ACCEPT (qp 4660", probe("127.0.0.1", port, "10.42.41.11", PROTOCOL_VERSION + 1))

    def test_tells_waiting_from_hanging_up(self):
        from tools.probe_imagemux import probe_idle, probe_magic
        waiting = self.fake_imagemux(speaks=PROTOCOL_VERSION)  # reads 64 bytes before it does anything
        self.assertIn("stays open", probe_idle("127.0.0.1", waiting, timeout=0.2))
        self.assertIn("stays open", probe_magic("127.0.0.1", waiting, "10.42.41.11", timeout=0.2))

        server = socket.socket()
        server.bind(("127.0.0.1", 0))
        server.listen(5)

        def hang_up():
            with server:
                for _ in range(2):
                    conn, _ = server.accept()
                    conn.close()

        threading.Thread(target=hang_up, daemon=True).start()
        port = server.getsockname()[1]
        self.assertIn("closed by imagemux", probe_idle("127.0.0.1", port))
        # The 4 bytes can reach a socket that is already closed, which answers with a reset.
        hung_up = probe_magic("127.0.0.1", port, "10.42.41.11")
        self.assertTrue("closed by imagemux" in hung_up or "reset by imagemux" in hung_up, hung_up)

    def test_unreachable_sender(self):
        from tools.probe_imagemux import probe
        server = socket.socket()
        server.bind(("127.0.0.1", 0))
        port = server.getsockname()[1]
        server.close()
        self.assertIn("cannot connect", probe("127.0.0.1", port, "10.42.41.11", PROTOCOL_VERSION))

    def test_describe(self):
        from tools.probe_imagemux import describe
        self.assertEqual(describe(b""), "nothing")
        self.assertIn("not a valid Accept", describe(bytes(64)))
        self.assertIn("5 bytes", describe(b"hello"))


if __name__ == "__main__":
    unittest.main()
