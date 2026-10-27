"""The receiver's C message parser (operators/cpp/common/fast_cbor_parse.c), compiled on the spot
and called through ctypes. Skipped when no C compiler is available."""

import ctypes
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "cpp/common/fast_cbor_parse.c"


@pytest.fixture(scope="module")
def fcp(tmp_path_factory):
    cc = shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        pytest.skip("no C compiler")
    lib = tmp_path_factory.mktemp("fcp") / "libfcp.so"
    subprocess.run([cc, "-O2", "-shared", "-fPIC", "-o", str(lib), str(SRC)], check=True)
    so = ctypes.CDLL(str(lib))
    u8p = ctypes.POINTER(ctypes.c_uint8)
    so.fast_extract_compressed_bslz4.argtypes = [
        ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(u8p), ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint64)]
    return so


def _text(s: str) -> bytes:
    assert len(s) < 24
    return bytes([0x60 | len(s)]) + s.encode()


def _image_message(timestamp: int, with_payload: bool = True) -> bytes:
    """A minimal Stream2 image message: type, image_id 300, a 64-bit timestamp, then the frame
    (tag 40 [[96, 96], tag 69 tag 56500 ["bslz4", 2, <12-byte bslz4 header>]])."""
    header = struct.pack(">QI", 96 * 96 * 2, 8192)
    payload = (b"\xd9\xdc\xb4\x83" + _text("bslz4") + b"\x02" + bytes([0x40 | len(header)])
               + header)
    frame = b"\xd8\x28\x82\x82\x18\x60\x18\x60\xd8\x45" + (payload if with_payload else b"")
    fields = (_text("type") + _text("image") + _text("image_id") + b"\x19\x01\x2c"
              + _text("real_time") + b"\x1b" + struct.pack(">Q", timestamp)
              + _text("data") + b"\xa1" + _text("threshold_1") + frame)
    return b"\xd9\xd9\xf7\xa4" + fields


def _extract(fcp, msg: bytes):
    u8p = ctypes.POINTER(ctypes.c_uint8)
    ptr, size, orig, elem, iid = (u8p(), ctypes.c_size_t(), ctypes.c_size_t(), ctypes.c_size_t(),
                                  ctypes.c_uint64())
    width, height = ctypes.c_uint32(), ctypes.c_uint32()
    rc = fcp.fast_extract_compressed_bslz4(msg, len(msg), ctypes.byref(ptr), ctypes.byref(size),
                                           ctypes.byref(orig), ctypes.byref(elem), ctypes.byref(width), ctypes.byref(height),
                                           ctypes.byref(iid))
    return rc, size.value, orig.value, elem.value, iid.value


def test_reads_an_image_message(fcp):
    assert _extract(fcp, _image_message(12345)) == (0, 12, 96 * 96 * 2, 2, 300)


def test_tag_bytes_inside_a_timestamp_are_skipped(fcp):
    """The DCU's timestamps contain the tag's three bytes about once per million frames; such a
    frame was rejected before 2.0.1 ("Expected array after Dectris tag, got 0x1a")."""
    msg = _image_message(0x00D9DCB41A000000)
    assert msg.index(b"\xd9\xdc\xb4") < msg.index(b"\xd8\x28")     # the false match comes first
    assert _extract(fcp, msg) == (0, 12, 96 * 96 * 2, 2, 300)


def test_a_message_without_payload_is_rejected(fcp):
    assert _extract(fcp, _image_message(0x00D9DCB41A000000, with_payload=False))[0] == -1
