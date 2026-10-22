# Imagemux RDMA transport package

`imx-receiver` provides the native libibverbs backend, imagemux wire protocol and TCP
handshake, registered GPU or mapped-host rings, and GPU dispatch helpers. The shared
[Holoscan pipeline](../docs/holoscan_basic_ingest.md) owns ZMQ/RDMA selection, runtime GPU
tensor allocation, raster placement, BF/ADF processing, configuration, and the container.

## Build the reusable wheel

From this directory:

```bash
python3 -m pip wheel --no-deps . -w dist
python3 -m pip install dist/imx_receiver-*.whl
```

Building requires Python 3.12 or later, CMake, a C/C++ compiler, and `libibverbs-dev`.
At runtime, install `libibverbs1` and `ibverbs-providers`; the wheel does not bundle them.
Protocol modules need NumPy and cbor2. GPU ring and dispatch helpers additionally need a
compatible CuPy installation. The unified CUDA 13 image installs these dependencies.

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
for the supported 96×96 uint16 application.

## Tests

`tests/test_imagemux.py` and `tests/test_messages.py` cover the protocol and geometry
without a GPU. GPU utility tests
are in `tests/test_gpu.py`; the shared application tests, including native RDMA loopback,
are under `holoscan_pipeline/tests`. Five existing probe tests reference the absent
`tools.probe_imagemux` helper and remain a separate pre-existing failure.
