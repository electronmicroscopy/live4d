"""Real-time STEM processing: a modular ingest → processing → stream-out pipeline.

`ingest` receives and decompresses detector frames (Holoscan C++ ops on the GPU at full rate, or
a pure-Python CPU path anywhere), `processing`
turns batches of frames into a reconstruction behind a small `Accumulator` contract, and
`streaming` publishes the reconstruction-so-far over ZeroMQ. Parallax / tcBF is the worked
example; new algorithms drop in behind the same contract (see `processing.registry`).
"""

__version__ = "3.0.0"
