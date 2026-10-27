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
# Wire formats of imagemux's RDMA transport (port of imagemux/src/rdma/wire.rs, protocol version 2).
#
# Two formats: the three TCP handshake messages (Hello, Accept, Ready) and the 128-byte frame header
# that precedes every frame in the ring. All integers are little endian.
# ---------------------------------------------------------------------------------------------

HANDSHAKE_MAGIC = int.from_bytes(b"IMXR", "little")
FRAME_MAGIC = int.from_bytes(b"IMXF", "little")
PROTOCOL_VERSION = 2

# Frames start at multiples of this ring offset; a ring's length is a multiple too.
FRAME_ALIGN = 128
FRAME_HEADER_LEN = 128

HELLO_LEN = 64
ACCEPT_LEN = 64
READY_LEN = 8


class HandshakeError(ValueError):
    pass


def padded(n: int) -> int:
    return -(-n // FRAME_ALIGN) * FRAME_ALIGN


def frame_span(data_size: int) -> int:
    """Ring bytes occupied by a frame with ``data_size`` bytes of data."""
    return padded(FRAME_HEADER_LEN + data_size)


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


def _check_preamble(buf: bytes):
    magic, version, _ = _PREAMBLE.unpack_from(buf)
    if magic != HANDSHAKE_MAGIC:
        raise HandshakeError("bad magic")
    if version != PROTOCOL_VERSION:
        raise HandshakeError(f"unsupported protocol version {version}")


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
        if length < 2 * FRAME_ALIGN or length % FRAME_ALIGN or addr % FRAME_ALIGN:
            raise HandshakeError("misaligned ring parameters")
        return cls(endpoint, addr, length, rkey)


@dataclass(frozen=True)
class Accept:
    """imagemux to receiver: imagemux's queue pair and the credit word we write positions to."""

    endpoint: QpEndpoint
    credit_addr: int
    credit_rkey: int

    def encode(self) -> bytes:
        e = self.endpoint
        buf = _PREAMBLE.pack(HANDSHAKE_MAGIC, PROTOCOL_VERSION, 0)
        buf += _ENDPOINT.pack(e.qp_num, e.psn, e.lid, e.mtu, e.gid)
        buf += struct.pack("<QI", self.credit_addr, self.credit_rkey)
        return buf.ljust(ACCEPT_LEN, b"\0")

    @classmethod
    def decode(cls, buf: bytes) -> "Accept":
        _check_preamble(buf)
        endpoint = QpEndpoint(*_ENDPOINT.unpack_from(buf, _PREAMBLE.size))
        addr, rkey = struct.unpack_from("<QI", buf, _PREAMBLE.size + _ENDPOINT.size)
        return cls(endpoint, addr, rkey)


def encode_ready() -> bytes:
    """Receiver to imagemux: QP connected and RECEIVEs posted, frames may be written."""
    return _PREAMBLE.pack(HANDSHAKE_MAGIC, PROTOCOL_VERSION, 0)


def decode_ready(buf: bytes):
    _check_preamble(buf)


# ---------------------------------------------------------------------------------------------
# Frame header

# magic, version, kind, sequence, position, data_size, series_id,
# image_id, batch_id, batch_size, dim_x, dim_y, dtype, compression, uncompressed_size,
# channel, channel_count
_FRAME_HEADER = struct.Struct("<IHHQQQQQQIIIHHQHH")


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
    sequence: int  # per-connection frame counter from 0; its low 32 bits are the write's immediate
    position: int  # logical stream position of this header; ring offset is position % ring_len
    data_size: int
    series_id: int = 0
    image: ImageHeader = field(default_factory=ImageHeader)

    def encode(self) -> bytes:
        i = self.image
        buf = _FRAME_HEADER.pack(FRAME_MAGIC, PROTOCOL_VERSION, self.kind, self.sequence, self.position,
                          self.data_size, self.series_id, i.image_id, i.batch_id, i.batch_size,
                          i.dim_x, i.dim_y, i.dtype, i.compression, i.uncompressed_size,
                          i.channel, i.channel_count)
        return buf.ljust(FRAME_HEADER_LEN, b"\0")

    @classmethod
    def decode(cls, buf) -> "FrameHeader":
        (magic, version, kind, sequence, position, data_size, series_id, image_id, batch_id,
         batch_size, dim_x, dim_y, dtype, compression, uncompressed_size, channel,
         channel_count) = _FRAME_HEADER.unpack_from(buf)
        if magic != FRAME_MAGIC:
            raise HandshakeError("bad magic")
        if version != PROTOCOL_VERSION:
            raise HandshakeError(f"unsupported protocol version {version}")
        image = ImageHeader(image_id, batch_id, batch_size, dim_x, dim_y,
                            _enum(DataType, dtype), _enum(Compression, compression),
                            uncompressed_size, channel, channel_count)
        return cls(_enum(FrameKind, kind), sequence, position, data_size, series_id, image)


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
# Every RDMA write with immediate announces one frame. The immediate is the frame's sequence number
# mod 2**32; the frame itself sits at the reader's current logical position, so the reader only
# tracks that position and checks the header it finds there. A WRAP frame carries no data: the
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
    """

    def __init__(self, ring_len: int, settle_s: float = 0.0):
        if ring_len < 2 * FRAME_ALIGN or ring_len % FRAME_ALIGN:
            raise ValueError(f"ring length {ring_len} must be a multiple of {FRAME_ALIGN}, "
                             f"at least {2 * FRAME_ALIGN}")
        self.ring_len = ring_len
        self.settle_s = settle_s
        self.position = 0  # logical position up to which frames have been handed out
        self.sequence = 0  # sequence number of the next frame

    def next(self, ring, imm: int) -> Frame | None:
        """Frame announced by immediate ``imm``; None for a wrap marker. ``ring`` is a buffer of
        ``ring_len`` bytes (bytes-like, indexable by slice)."""
        if imm != self.sequence & 0xFFFFFFFF:
            raise HandshakeError(f"frame announced out of order: immediate {imm}, "
                            f"expected {self.sequence & 0xFFFFFFFF}")
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
                header = FrameHeader.decode(raw)
                if header.sequence == self.sequence and header.position == self.position:
                    return header
                problem = (f"frame header says sequence {header.sequence} position "
                           f"{header.position}, expected sequence {self.sequence} position "
                           f"{self.position}")
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
    """What we tell imagemux: where to write frames, and our queue pair."""

    ring: RemoteRegion
    endpoint: QpEndpoint = QpEndpoint()


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
    """imagemux's handshake: Hello (64 B) -> Accept (64 B) -> Ready (8 B)."""

    def run(self, sock: socket.socket, local: LocalInfo) -> Peer:
        """Send our ring and queue pair; return imagemux's credit word and queue pair."""
        r = local.ring
        sock.sendall(Hello(local.endpoint, r.addr, r.length, r.rkey).encode())
        try:
            accept = Accept.decode(_read_exact(sock, ACCEPT_LEN))
        except ConnectionError as e:
            raise ConnectionError(
                "imagemux closed the connection after our Hello without sending its Accept. It "
                "decides that on its side: check imagemux's log (it rejects a bad Hello, e.g. a "
                "misaligned ring or a protocol version mismatch, and it fails if it cannot set up "
                "its own RDMA queue pair or buffers; a stale state from an earlier failed "
                "connection also needs a restart of the sender)") from e
        return Peer(RemoteRegion(accept.credit_addr, 8, accept.credit_rkey), accept.endpoint)

    def ready(self, sock: socket.socket) -> None:
        """Tell imagemux our QP is connected and RECEIVEs are posted (after ``attach_qp``)."""
        sock.sendall(encode_ready())


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
