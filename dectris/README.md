# Imagemux RDMA transport package

`imx-receiver` provides the native libibverbs backend, imagemux wire protocol and TCP
handshake, registered GPU or mapped-host rings, and GPU dispatch helpers. The shared
[live4d application](../docs/holoscan_basic_ingest.md) owns ZMQ/RDMA selection, runtime GPU
tensor allocation, raster placement, BF/ADF processing, configuration, and the container.

## Build the reusable wheel

From this directory:

```bash
python3 -m pip wheel --no-deps . -w dist
python3 -m pip install dist/imx_receiver-*.whl
```

Building on Linux requires Python 3.12 or later with development headers, CMake 3.18 or
later, a C++20-capable C/C++ compiler, `pkg-config`, and `libibverbs-dev`.
At runtime, install `libibverbs1` and `ibverbs-providers`; the wheel does not bundle them.
Protocol modules need NumPy and cbor2. GPU ring and dispatch helpers additionally need a
compatible CuPy installation. The unified CUDA 13 image installs these dependencies.

The distribution is named `imx-receiver`, but its Python import is `dectris`.
`dectris-compression`, used by the earlier Python experiments, is a separate distribution;
it does not supply this package's RDMA backend. Building this wheel does not build the
Live4D operators or generated schemas; use the unified container for the application.

## Components

| Component | Public interfaces |
| --- | --- |
| Native verbs binding | `imxrdma.Context`, `QueuePair`, `MemoryRegion` |
| Connection and ownership | `VerbsConn`, `watch_liveness` |
| TCP handshake | `TcpConnector`, `ImagemuxCodec`, `LocalInfo`, `Link`, `Peer` |
| Frame protocol | `FrameReader`, `FrameHeader`, `FrameKind`, `RemoteRegion`, `QpEndpoint` |
| Cumulative credits | `CreditTracker` |
| GPU utilities | `Dispatcher`, `DeviceRing` |
| Series and scan geometry | `Series`, `Start`, `Batch`, `End`, `ScanShape`, `scan_positions` |

Imports are lazy so protocol tools can run without GPU libraries. Keep the `VerbsConn`,
handshake link, and queue-pair owner alive until all GPU reads and the runtime session have
finished. `QueuePair.native_handle` is a borrowed native pointer for integration with the
shared C++ source operator; it does not transfer ownership.

`dectris.connection.RegisteredRing` separates ring allocation/registration from QP
lifetime. Multiple connections can reuse it after prior NIC writes and GPU reads drain.
The [Live4D acquisition controller](../docs/holoscan_basic_ingest.md#rdma-acquisition-routing)
uses this ownership model for one or more GPUs.

## Protocol and flow control

The receiver connects to the sender's imagemux TCP handshake endpoint. Hello/Accept/Ready
exchange QP parameters, receiver ring address/length/rkey, and the sender's credit-word
address/rkey. The socket stays open as a liveness signal. No `rdma_cm` connection manager
is used.

The sender writes a 128-byte frame header and payload into the registered ring. The final
write uses WRITE_WITH_IMM; its sequence number identifies the completed frame. Frames are
128-byte aligned. START and END delimit acquisitions; IMAGE frames contain consecutive
image IDs; WRAP skips the remaining bytes before returning to offset zero. The receiver
checks logical position and sequence and replenishes the zero-length receives consumed by
immediate completions.

The receiver RDMA-writes an eight-byte cumulative consumed position into the sender's
credit word. The sender may reuse ring bytes only after this position advances past them.
GPU readers must finish before credit is returned. A frame split across output tensors
therefore retains its ring bytes until every piece has been gathered. Credits for earlier
complete frames can advance while a later output batch is still being assembled.

The ingress ring and the downstream tensor pool are separate allocations. `gpu` rings
require working GPUDirect RDMA support; `host_mapped` rings use registered mapped pinned
host memory. Both paths require sufficient locked-memory limits and verbs device access.
Use the [shared build and run instructions](../docs/holoscan_basic_ingest.md#build-and-run)
for the application, which supports 96×96 (2×2 binned) and 192×192 (unbinned, 1×1)
uint16 images. START metadata supplies the series dimensions; RDMA IMAGE headers must
match them. The standalone `Dispatcher.dispatch()` accepts the series shape through
its `shape` argument and defaults to `(96, 96)`.

## Tests

From the repository root, protocol and geometry tests can run directly from source with
NumPy, cbor2, and pytest, without building the native wheel or installing GPU libraries:

```bash
python3 -m pip install numpy cbor2 pytest
PYTHONPATH=dectris/src python3 -m pytest -q \
  dectris/tests/test_imagemux.py dectris/tests/test_messages.py -k 'not ProbeTest'
```

The explicit exclusion covers five `ProbeTest` cases importing the absent
`tools.probe_imagemux` helper. The unfiltered protocol suite therefore has a known failure;
this command validates the remaining protocol/geometry coverage only. Restore that helper
or retire its tests in a separate code change rather than treating them as passing.

`tests/test_gpu.py` requires CUDA and CuPy. `tests/test_imxrdma.py` exercises the native
verbs backend and requires the built extension plus `IMXRDMA_LOCAL_ADDR` set to an active
RoCE address; its GPU case also requires CUDA. Shared application tests, including RDMA
batching loopback, are under `holoscan_pipeline/tests`; follow the
[integration-test instructions](../docs/holoscan_basic_ingest.md#gpu-and-rdma-integration-checks)
for their additional `RDMA_TEST_IN_IMAGE=1` opt-in.
