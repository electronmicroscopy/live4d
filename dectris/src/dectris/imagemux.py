"""imagemux's RDMA transport: wire formats, protocol helpers, frame reader and connection.

No GPU imports, so it can be used and tested without a GPU.
"""

import socket
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable


# ---------------------------------------------------------------------------------------------
# Wire formats
#
# Wire formats of imagemux's RDMA transport (port of imagemux/src/rdma/wire.rs, protocol versions
# 2 and 3).
#
# Two formats: the three TCP handshake messages (Hello, Accept, Ready) and the 128-byte frame header
# that precedes every frame in a ring. All integers are little endian. Protocol 2 has one ring.
# Protocol 3 lets the receiver advertise up to 256 rings that imagemux fills with series round
# robin; the frame header and the immediate then also carry the ring index.
# ---------------------------------------------------------------------------------------------

HANDSHAKE_MAGIC = int.from_bytes(b"IMXR", "little")
FRAME_MAGIC = int.from_bytes(b"IMXF", "little")
PROTOCOL_VERSION = 2  # single-ring protocol, the default
MULTI_RING_VERSION = 3
PROTOCOL_VERSIONS = (PROTOCOL_VERSION, MULTI_RING_VERSION)

# Frames start at multiples of this ring offset; a ring's length is a multiple too.
FRAME_ALIGN = 128
FRAME_HEADER_LEN = 128

HELLO_LEN = 64  # protocol 3: the fixed part, followed by a ring descriptor per ring
RING_DESCRIPTOR_LEN = 24
ACCEPT_LEN = 64
READY_LEN = 8

# Protocol 3 immediates: ring index in the upper 8 bits, low 24 bits of the ring's sequence below.
RING_BITS = 8
MAX_RINGS = 1 << RING_BITS
SEQUENCE_BITS = 32 - RING_BITS
SEQUENCE_MASK = (1 << SEQUENCE_BITS) - 1


class HandshakeError(ValueError):
    pass


def padded(n: int) -> int:
    return -(-n // FRAME_ALIGN) * FRAME_ALIGN


def frame_span(data_size: int) -> int:
    """Ring bytes occupied by a frame with ``data_size`` bytes of data."""
    return padded(FRAME_HEADER_LEN + data_size)


def immediate(ring: int, sequence: int, version: int = PROTOCOL_VERSION) -> int:
    """The immediate announcing frame ``sequence`` of ``ring``."""
    if version == PROTOCOL_VERSION:
        return sequence & 0xFFFFFFFF
    return (ring << SEQUENCE_BITS) | (sequence & SEQUENCE_MASK)


def split_immediate(imm: int) -> tuple[int, int]:
    """Protocol 3: the ring index and the low sequence bits an immediate carries."""
    return imm >> SEQUENCE_BITS, imm & SEQUENCE_MASK


class FrameKind(IntEnum):
    START = 1  # Stream2 start message (CBOR)
    IMAGE = 2  # raw image data of one channel of one or more images
    END = 3  # Stream2 end message (CBOR)
    WRAP = 0xFFFF  # no payload; the writer continues at ring offset 0


class DataType(IntEnum):
    NONE = 0
    UINT8 = 1
    UINT16 = 2
    UINT32 = 3
    UINT64 = 4

    @property
    def elem_size(self) -> int:
        return (0, 1, 2, 4, 8)[self]


class Compression(IntEnum):
    NONE = 0
    BSLZ4 = 1
    BSZSTD = 2
    LZ4 = 3
    TRPX = 4


def _enum(cls, value: int):
    try:
        return cls(value)
    except ValueError:
        raise HandshakeError(f"unsupported {cls.__name__.lower()} {value}") from None


# ---------------------------------------------------------------------------------------------
# Handshake

_ENDPOINT = struct.Struct("<IIHBx16s")  # qp_num, psn, lid, mtu, pad, gid
_PREAMBLE = struct.Struct("<IHH")  # magic, version, reserved


@dataclass(frozen=True)
class QpEndpoint:
    """Queue pair parameters a peer needs to connect its RC queue pair to ours."""

    qp_num: int = 0
    psn: int = 0
    lid: int = 0
    mtu: int = 0  # enum ibv_mtu: 1 = 256 bytes ... 5 = 4096 bytes
    gid: bytes = bytes(16)


def _check_preamble(buf: bytes, expected: int = PROTOCOL_VERSION):
    magic, version, _ = _PREAMBLE.unpack_from(buf)
    if magic != HANDSHAKE_MAGIC:
        raise HandshakeError("bad magic")
    if version != expected:
        raise HandshakeError(f"unsupported protocol version {version}, expected {expected}")


def _check_ring(addr: int, length: int):
    if length < 2 * FRAME_ALIGN or length % FRAME_ALIGN or addr % FRAME_ALIGN:
        raise HandshakeError("misaligned ring parameters")


@dataclass(frozen=True)
class Hello:
    """Receiver to imagemux: our queue pair and the ring imagemux may write frames into."""

    endpoint: QpEndpoint
    ring_addr: int
    ring_len: int
    ring_rkey: int

    def encode(self) -> bytes:
        e = self.endpoint
        buf = _PREAMBLE.pack(HANDSHAKE_MAGIC, PROTOCOL_VERSION, 0)
        buf += _ENDPOINT.pack(e.qp_num, e.psn, e.lid, e.mtu, e.gid)
        buf += struct.pack("<QQI", self.ring_addr, self.ring_len, self.ring_rkey)
        return buf.ljust(HELLO_LEN, b"\0")

    @classmethod
    def decode(cls, buf: bytes) -> "Hello":
        _check_preamble(buf)
        endpoint = QpEndpoint(*_ENDPOINT.unpack_from(buf, _PREAMBLE.size))
        addr, length, rkey = struct.unpack_from("<QQI", buf, _PREAMBLE.size + _ENDPOINT.size)
        _check_ring(addr, length)
        return cls(endpoint, addr, length, rkey)


_RING_COUNT_OFFSET = _PREAMBLE.size + _ENDPOINT.size  # u16 at offset 36
_RING_DESCRIPTOR = struct.Struct("<QQI4x")  # address, length, rkey, reserved


@dataclass(frozen=True)
class MultiRingHello:
    """Protocol 3, receiver to imagemux: our queue pair and 1 to 256 rings for the series."""

    endpoint: QpEndpoint
    rings: "tuple[RemoteRegion, ...]"

    def encode(self) -> bytes:
        if not 1 <= len(self.rings) <= MAX_RINGS:
            raise HandshakeError(f"unsupported ring count {len(self.rings)} (1 to {MAX_RINGS})")
        e = self.endpoint
        buf = _PREAMBLE.pack(HANDSHAKE_MAGIC, MULTI_RING_VERSION, 0)
        buf += _ENDPOINT.pack(e.qp_num, e.psn, e.lid, e.mtu, e.gid)
        buf += struct.pack("<H", len(self.rings))
        buf = buf.ljust(HELLO_LEN, b"\0")
        for ring in self.rings:
            buf += _RING_DESCRIPTOR.pack(ring.addr, ring.length, ring.rkey)
        return buf

    @staticmethod
    def encoded_len(prefix: bytes) -> int:
        """Length of the whole message, given its fixed ``HELLO_LEN``-byte part."""
        _check_preamble(prefix, MULTI_RING_VERSION)
        count, = struct.unpack_from("<H", prefix, _RING_COUNT_OFFSET)
        if not 1 <= count <= MAX_RINGS:
            raise HandshakeError(f"unsupported ring count {count} (1 to {MAX_RINGS})")
        return HELLO_LEN + count * RING_DESCRIPTOR_LEN

    @classmethod
    def decode(cls, buf: bytes) -> "MultiRingHello":
        if len(buf) < HELLO_LEN:
            raise HandshakeError(f"handshake message of {len(buf)} bytes, expected {HELLO_LEN}")
        length = cls.encoded_len(buf[:HELLO_LEN])
        if len(buf) != length:
            raise HandshakeError(f"handshake message of {len(buf)} bytes, expected {length}")
        endpoint = QpEndpoint(*_ENDPOINT.unpack_from(buf, _PREAMBLE.size))
        rings = []
        for offset in range(HELLO_LEN, length, RING_DESCRIPTOR_LEN):
            addr, ring_len, rkey = _RING_DESCRIPTOR.unpack_from(buf, offset)
            _check_ring(addr, ring_len)
            rings.append(RemoteRegion(addr, ring_len, rkey))
        return cls(endpoint, tuple(rings))


@dataclass(frozen=True)
class Accept:
    """imagemux to receiver: imagemux's queue pair and the credit word we write positions to.

    Protocol 3 has one 8-byte credit word per ring, that of ring i at ``credit_addr + 8 * i``."""

    endpoint: QpEndpoint
    credit_addr: int
    credit_rkey: int
    version: int = PROTOCOL_VERSION

    def encode(self) -> bytes:
        e = self.endpoint
        buf = _PREAMBLE.pack(HANDSHAKE_MAGIC, self.version, 0)
        buf += _ENDPOINT.pack(e.qp_num, e.psn, e.lid, e.mtu, e.gid)
        buf += struct.pack("<QI", self.credit_addr, self.credit_rkey)
        return buf.ljust(ACCEPT_LEN, b"\0")

    @classmethod
    def decode(cls, buf: bytes, version: int = PROTOCOL_VERSION) -> "Accept":
        _check_preamble(buf, version)
        endpoint = QpEndpoint(*_ENDPOINT.unpack_from(buf, _PREAMBLE.size))
        addr, rkey = struct.unpack_from("<QI", buf, _PREAMBLE.size + _ENDPOINT.size)
        return cls(endpoint, addr, rkey, version)


def encode_ready(version: int = PROTOCOL_VERSION) -> bytes:
    """Receiver to imagemux: QP connected and RECEIVEs posted, frames may be written."""
    return _PREAMBLE.pack(HANDSHAKE_MAGIC, version, 0)


def decode_ready(buf: bytes, version: int = PROTOCOL_VERSION):
    _check_preamble(buf, version)


# ---------------------------------------------------------------------------------------------
# Frame header

# magic, version, kind, sequence, position, data_size, series_id,
# image_id, batch_id, batch_size, dim_x, dim_y, dtype, compression, uncompressed_size,
# channel, channel_count, ring (protocol 3; reserved and zero in protocol 2)
_FRAME_HEADER = struct.Struct("<IHHQQQQQQIIIHHQHHH")


@dataclass(frozen=True)
class ImageHeader:
    """Describes the data of an IMAGE frame: ``batch_size`` images of one channel with consecutive
    ids from ``image_id``, back to back (image i at ``i * dim_x * dim_y * elem_size`` when
    uncompressed)."""

    image_id: int = 0
    batch_id: int = 0
    batch_size: int = 0
    dim_x: int = 0
    dim_y: int = 0
    dtype: DataType = DataType.NONE
    compression: Compression = Compression.NONE
    uncompressed_size: int = 0
    channel: int = 0
    channel_count: int = 0


@dataclass(frozen=True)
class FrameHeader:
    kind: FrameKind
    sequence: int  # per-ring frame counter from 0; announced by the write's immediate
    position: int  # logical position of this header in its ring; ring offset is position % ring_len
    data_size: int
    series_id: int = 0
    image: ImageHeader = field(default_factory=ImageHeader)
    ring: int = 0  # protocol 3: index of the ring in the Hello
    version: int = PROTOCOL_VERSION

    def encode(self) -> bytes:
        if self.version == PROTOCOL_VERSION and self.ring:
            raise ValueError("protocol 2 frames have no ring index")
        i = self.image
        buf = _FRAME_HEADER.pack(FRAME_MAGIC, self.version, self.kind, self.sequence, self.position,
                          self.data_size, self.series_id, i.image_id, i.batch_id, i.batch_size,
                          i.dim_x, i.dim_y, i.dtype, i.compression, i.uncompressed_size,
                          i.channel, i.channel_count, self.ring)
        return buf.ljust(FRAME_HEADER_LEN, b"\0")

    @classmethod
    def decode(cls, buf, version: int = PROTOCOL_VERSION) -> "FrameHeader":
        (magic, found, kind, sequence, position, data_size, series_id, image_id, batch_id,
         batch_size, dim_x, dim_y, dtype, compression, uncompressed_size, channel,
         channel_count, ring) = _FRAME_HEADER.unpack_from(buf)
        if magic != FRAME_MAGIC:
            raise HandshakeError("bad magic")
        if found != version:
            raise HandshakeError(f"unsupported protocol version {found}, expected {version}")
        image = ImageHeader(image_id, batch_id, batch_size, dim_x, dim_y,
                            _enum(DataType, dtype), _enum(Compression, compression),
                            uncompressed_size, channel, channel_count)
        if version == PROTOCOL_VERSION:
            ring = 0  # reserved bytes in protocol 2
        return cls(_enum(FrameKind, kind), sequence, position, data_size, series_id, image, ring,
                   found)


# ---------------------------------------------------------------------------------------------
# Names and helpers
#
# The remote region descriptor and the credit word, shared by the imagemux receive path.
# The wire formats are above, in the "Wire formats" section.
# ---------------------------------------------------------------------------------------------

CREDIT_BYTES = 8  # the credit word: the consumed logical ring position, a little-endian u64


@dataclass(frozen=True)
class RemoteRegion:
    """What a peer needs to RDMA-write into a region: address, length, rkey."""

    addr: int
    length: int
    rkey: int


class CreditTracker:
    """The ring position that may be returned to imagemux.

    Work that reads the ring (kernels on the GPUs) finishes asynchronously, and imagemux may
    overwrite the ring below a position as soon as it gets it as credit. ``add`` registers the
    work up to a position with the CUDA events marking its end; ``poll`` returns the highest
    position all work below which has finished. Positions are credited in order.
    """

    def __init__(self):
        self.pending = deque()  # (position, events, keep) in order of position
        self.safe = 0

    def add(self, position: int, events=(), keep=()):
        """Everything up to ``position`` is read by work that ends with ``events``; ``keep``
        (host memory the work reads) is held until then."""
        self.pending.append((position, list(events), list(keep)))

    def poll(self) -> int:
        while self.pending and all(event.done for event in self.pending[0][1]):
            self.safe = self.pending.popleft()[0]
        return self.safe


# ---------------------------------------------------------------------------------------------
# Frame reader
#
# Locates the frames imagemux wrote into the ring (Python counterpart of the Rust FrameReader).
#
# Every RDMA write with immediate announces one frame. In protocol 2 the immediate is the frame's
# sequence number mod 2**32; in protocol 3 it is ``ring << 24 | sequence mod 2**24``, and each ring
# has its own reader. The frame itself sits at the reader's current logical position, so the reader
# only tracks that position and checks the header it finds there. A WRAP frame carries no data: the
# writer skipped the rest of the ring and continues at offset 0.
#
# Port of imagemux/src/rdma/reader.rs.
#
# No holoscan / RDMA imports.
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Frame:
    header: FrameHeader
    data_offset: int  # ring offset of the first data byte; the data does not wrap


class FrameReader:
    """Tracks the logical position and sequence number of the next frame in the ring.

    ``settle_s``: how long to keep re-reading a header that is not (yet) the expected one before
    giving up. 0 for a ring in host memory, where the data is there once the completion is. For a
    ring in GPU memory the NIC's write may become visible a little after its completion.
    ``version`` 3 reads ring ``ring`` of a multi-ring receiver.
    """

    def __init__(self, ring_len: int, settle_s: float = 0.0, *, version: int = PROTOCOL_VERSION,
                 ring: int = 0):
        if ring_len < 2 * FRAME_ALIGN or ring_len % FRAME_ALIGN:
            raise ValueError(f"ring length {ring_len} must be a multiple of {FRAME_ALIGN}, "
                             f"at least {2 * FRAME_ALIGN}")
        if version not in PROTOCOL_VERSIONS or not 0 <= ring < MAX_RINGS or (
                version == PROTOCOL_VERSION and ring):
            raise ValueError(f"invalid ring {ring} for protocol {version}")
        self.ring_len = ring_len
        self.settle_s = settle_s
        self.version = version
        self.ring = ring
        self.position = 0  # logical position up to which frames have been handed out
        self.sequence = 0  # sequence number of the next frame

    def next(self, ring, imm: int) -> Frame | None:
        """Frame announced by immediate ``imm``; None for a wrap marker. ``ring`` is a buffer of
        ``ring_len`` bytes (bytes-like, indexable by slice)."""
        expected = immediate(self.ring, self.sequence, self.version)
        if imm != expected:
            raise HandshakeError(f"frame announced out of order: immediate {imm:#x}, "
                                 f"expected {expected:#x}")
        offset = self.position % self.ring_len
        header = self._header(ring, offset)
        self.sequence += 1
        if header.kind == FrameKind.WRAP:
            self.position += self.ring_len - offset
            return None
        span = frame_span(header.data_size)
        if offset + span > self.ring_len:
            raise HandshakeError(f"frame at position {self.position} runs past the end of the ring")
        self.position += span
        return Frame(header, offset + FRAME_HEADER_LEN)

    def _header(self, ring, offset: int) -> FrameHeader:
        """The header at ``offset`` if it is the expected frame's; re-read for ``settle_s``."""
        deadline = time.monotonic() + self.settle_s
        while True:
            raw = ring[offset:offset + FRAME_HEADER_LEN]
            try:
                header = FrameHeader.decode(raw, self.version)
                if (header.sequence == self.sequence and header.position == self.position
                        and header.ring == self.ring):
                    return header
                problem = (f"frame header says ring {header.ring} sequence {header.sequence} "
                           f"position {header.position}, expected ring {self.ring} sequence "
                           f"{self.sequence} position {self.position}")
            except HandshakeError as e:
                problem = str(e)
            if time.monotonic() >= deadline:
                raise HandshakeError(f"{problem} (ring offset {offset}, first bytes "
                                f"{bytes(raw[:32]).hex()})")
            time.sleep(2e-5)


# ---------------------------------------------------------------------------------------------
# Connection to imagemux
#
# How the receiver gets connected to imagemux, and how it notices imagemux has gone.
#
# imagemux uses plain libibverbs (no librdmacm): the receiver connects to its TCP handshake port for
# a three-message handshake (Hello, Accept, Ready; see "Wire formats"), and that socket then
# only signals liveness. The queue pair is brought up out of band (``attach_qp``).
#
# Everything the pipeline needs from setup is a ``Link``: our connected queue pair, imagemux's
# credit word and queue pair, and a liveness flag. ``TcpConnector`` produces one.
#
# No holoscan / RDMA imports here, so this module can be tested on its own.
# ---------------------------------------------------------------------------------------------

HANDSHAKE_PORT = 31003  # stream_v2 port + 2


@dataclass(frozen=True)
class LocalInfo:
    """What we tell imagemux: where to write frames, and our queue pair.

    Protocol 3 advertises ``rings``; when empty, ``ring`` is the only ring."""

    ring: RemoteRegion
    endpoint: QpEndpoint = QpEndpoint()
    rings: tuple[RemoteRegion, ...] = ()


@dataclass(frozen=True)
class Peer:
    """What imagemux tells us: where to write credits, and its queue pair."""

    credit: RemoteRegion
    endpoint: QpEndpoint | None = None


class Liveness:
    """Watches an idle TCP socket; ``alive`` clears when imagemux closes it or it errors."""

    def __init__(self, sock: socket.socket | None = None):
        self._sock = sock
        self._dead = threading.Event()
        self.reason = ""
        if sock is not None:
            threading.Thread(target=self._watch, name="imagemux-liveness", daemon=True).start()

    @property
    def alive(self) -> bool:
        return not self._dead.is_set()

    def mark_dead(self, reason: str):
        if not self._dead.is_set():
            self.reason = reason
            self._dead.set()

    def wait_dead(self, timeout: float | None = None) -> bool:
        return self._dead.wait(timeout)

    def _watch(self):
        try:
            self._sock.settimeout(None)
            while self._sock.recv(4096):
                pass  # liveness only: anything sent after the handshake is ignored
            self.mark_dead("imagemux closed the handshake connection")
        except OSError as e:
            self.mark_dead(f"handshake connection error: {e}")


@dataclass
class Link:
    conn: Any  # what attach_qp returned: the connected queue pair (imxrdma.QueuePair)
    peer: Peer
    liveness: Liveness
    close: Callable[[], None] = lambda: None


class ImagemuxCodec:
    """imagemux's handshake: Hello (64 B, plus 24 B per ring in protocol 3) -> Accept (64 B) ->
    Ready (8 B)."""

    def __init__(self, version: int = PROTOCOL_VERSION):
        if version not in PROTOCOL_VERSIONS:
            raise ValueError(f"unsupported protocol version {version}")
        self.version = version

    def run(self, sock: socket.socket, local: LocalInfo) -> Peer:
        """Send our rings and queue pair; return imagemux's credit words and queue pair.

        In protocol 3 the returned credit region spans one 8-byte word per ring."""
        rings = local.rings or (local.ring,)
        if self.version == PROTOCOL_VERSION:
            if len(rings) != 1:
                raise ValueError("protocol 2 supports exactly one ring")
            r = rings[0]
            sock.sendall(Hello(local.endpoint, r.addr, r.length, r.rkey).encode())
        else:
            sock.sendall(MultiRingHello(local.endpoint, tuple(rings)).encode())
        try:
            accept = Accept.decode(_read_exact(sock, ACCEPT_LEN), self.version)
        except ConnectionError as e:
            raise ConnectionError(
                "imagemux closed the connection after our Hello without sending its Accept. It "
                "decides that on its side: check imagemux's log (it rejects a bad Hello, e.g. a "
                "misaligned ring or a protocol version mismatch, and it fails if it cannot set up "
                "its own RDMA queue pair or buffers; a stale state from an earlier failed "
                "connection also needs a restart of the sender)") from e
        return Peer(RemoteRegion(accept.credit_addr, CREDIT_BYTES * len(rings), accept.credit_rkey),
                    accept.endpoint)

    def ready(self, sock: socket.socket) -> None:
        """Tell imagemux our QP is connected and RECEIVEs are posted (after ``attach_qp``)."""
        sock.sendall(encode_ready(self.version))


def _read_exact(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("imagemux closed the connection during the handshake")
        buf += chunk
    return buf


class TcpConnector:
    """TCP client on imagemux's handshake port, then the QP bring-up.

    ``local()`` returns our queue pair endpoint with the ring. ``attach_qp(peer) -> conn``
    connects that QP to ``peer.endpoint`` and posts RECEIVEs (``VerbsConn.connect`` in
    connection.py); only then is Ready sent.
    """

    def __init__(self, addr: str, codec: ImagemuxCodec, local: Callable[[], LocalInfo],
                 attach_qp: Callable[[Peer], Any], port: int = HANDSHAKE_PORT,
                 timeout_s: float = 10.0):
        self.addr, self.port, self.codec = addr, port, codec
        self.local, self.attach_qp, self.timeout_s = local, attach_qp, timeout_s

    def connect(self) -> Link:
        sock = socket.create_connection((self.addr, self.port), timeout=self.timeout_s)
        try:
            peer = self.codec.run(sock, self.local())
            conn = self.attach_qp(peer)
            self.codec.ready(sock)
        except BaseException:
            sock.close()
            raise
        def close():
            # Wake the liveness thread's blocking recv before closing the socket.
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()

        return Link(conn, peer, Liveness(sock), close)
