"""Live scans: each detector series received by live4d, ANS-encoded on the GPU as it arrives (quantem.gpu)."""

from .scans import LiveScans

__all__ = ["LiveFrames", "LiveScans", "ReceiverBusy"]


def __getattr__(name):
    # LiveFrames needs Holoscan and the compiled operators; LiveScans (reading saved scans) does not.
    if name in ("LiveFrames", "ReceiverBusy"):
        from . import receiver
        return getattr(receiver, name)
    raise AttributeError(name)
