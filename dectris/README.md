# DECTRIS RDMA Example

Consume DECTRIS streams delivered with RDMA into 
a Holoscan pipeline directly as GPU tensors. The image series, bookmarked by START and an END frames, is processed on the GPU of the ring they were written to. The images are
emitted as 4D-STEM tensors with their scan positions (see [4D-STEM output](#4d-stem-output)), and
the example builds virtual detector images from them.

```
imagemux -> gpu frame ring -> RdmaRx -> ScanTensor -> VirtualDetector
   ^------------- consumed position (RDMA WRITE) ---^
```

## Prerequisites

This example builds on the NGC Holoscan SDK 5.0.0 CUDA 13 container image. Log in to the NGC container registry to pull
the Holoscan image if not already available:
```
docker login nvcr.io
```
Username `$oauthtoken`, password your NGC API key.

## Python wheel

The receiver is provided as a python package: the Holoscan operators
(`RdmaRxOp`), wire protocol (`imagemux`, `messages`), GPU dispatch
(`dispatch`), connection helpers (`connection`), wrapping a native (C/C++) module `imxrdma`. The
receiver app itself is `examples/pipeline/app.py`, and imports all of this from the wheel. The 
`pyproject.toml` builds it with scikit-build-core which runs `native/CMakeLists.txt`:
```
python3 -m pip wheel --no-deps . -w dist     # needs cmake, a C/C++ compiler, libibverbs-dev
python3 -m pip install dist/imx_receiver-*.whl
```
The Dockerfile does this automatically in a build stage and installs only the wheel into the container image. The result
is one wheel per CPython version and architecture, e.g.
`imx_receiver-0.1.0-cp312-cp312-linux_aarch64.whl` for the Holoscan SDK 5 image.

- **Run time:** The wheel links libibverbs dynamically; the machine needs
  rdma-core (`libibverbs1`, `ibverbs-providers`), Holoscan SDK 5, and CuPy. The receiver image
  installs the CUDA 13 CuPy extra; use `pip install "imx-receiver[cuda13]"` when installing
  the wheel into another compatible environment.

## Base image

Other containers can build their own Holoscan apps and operators on `imx-receiver-base`: the
Holoscan image with rdma-core and the wheel installed, and the wheel file itself
(in `/opt/dectris/stanford-dectris-nvidia-4d-stem/wheels`). This is the `imx-base`
target of the Dockerfile; the example image of this repo is built on it.
```
dectris/tools/build_base_image.sh    # imx-receiver-base:0.1.0 and :latest
```
A consumer:
```
FROM imx-receiver-base:0.1.0
COPY app.py .
```

Everything from `dectris`

| What | Names |
|---|---|
| Source operator | `RdmaRxOp`, `RingPath` |
| 4D-STEM tensors | `ScanTensorOp`, `ScanShape`, `Segment`, `scan_positions` |
| Setup and running | `VerbsConn`, `TcpConnector`, `ImagemuxCodec`, `LocalInfo`, `watch_liveness` |
| Passed between operators | `Start`, `Batch`, `End`, `Series`, `IMAGE_SHAPE`, `IMAGE_BYTES` |
| Protocol | `FrameReader`, `FrameHeader`, `FrameKind`, `CreditTracker`, `Peer`, `Link`, `RemoteRegion`, `QpEndpoint` |
| GPU | `Dispatcher`, `DeviceRing` |
| RDMA verbs | `imxrdma` (`Context`, `QueuePair`, `MemoryRegion`) |

`RdmaRxOp` emits on `output{i}` (per ring) and the app's own operators take those on an input.


In `examples/pipeline/` is an example app: `app.py` wires `RdmaRxOp` into
`ScanTensorOp` and the operator defined in `operators.py` (`VirtualDetectorOp`, which builds bright-field and annular dark-field images of the scan). It also has a container of its own based on the base image, with `app.py` as entrypoint; its configuration is in `examples/pipeline/pipeline.yaml`:
```
examples/pipeline/run.sh
```
`run.sh` builds `imx-pipeline-example` and runs it with the RDMA device,
GPU, host network and memlock flags.

An image for another example can take just the wheel:
`COPY --from=imx-receiver-base:0.1.0 /opt/dectris/stanford-dectris-nvidia-4d-stem/wheels /tmp/wheels`
then `pip install /tmp/wheels/imx_receiver-*.whl` (cp312, aarch64, with rdma-core installed).

## Running

```
./examples/pipeline/run.sh --config ./examples/pipeline/pipeline.yaml
```
`run.sh` builds the `imx-pipeline-example` image (`BUILD=0` skips that) and runs `app.py` in it
with the RDMA device, GPU, host network and memlock flags; its arguments are passed on to
`app.py`.


## Configuration

The settings are read from YAML and validated before the graph is compiled. The default file is
`examples/pipeline/pipeline.yaml`, which is baked into the image. To use another file, pass it as
the first argument (`run.sh` mounts it into the container):
```
./examples/pipeline/run.sh --config my.yaml
```


| Key | Default | Meaning |
|---|---|---|
| `connection.remote_address` | required | imagemux's RDMA handshake address, e.g. `10.42.41.10:31003` |
| `connection.local_address` | required | address of the RoCE interface to use |
| `connection.device` | `""` | RDMA device; `""` holding `local_address` |
| `connection.ring_gpu` | 0 | GPU processing the images |
| `connection.ring_memory` | `gpu` | ring allocation: `gpu` for GPUDirect RDMA, or `host_mapped` for pinned host memory with a CUDA alias |
| `connection.ring_mib` | 64 | ring size in MiB |
| `connection.timeout` | 10 | seconds for the TCP connection to imagemux |
| `execute.seconds` | 10 | run time, 0 = until Ctrl+C |
| `execute.threads` | 3 | execution partition layout: 1 groups all operators, 2 separates RX from processing, 3 or more separates all three stages |
| `receiver.receive_only` | false | no GPU work: frames are counted and their credit returned at once; the series end line then shows the raw rate the sender delivers |
| `receiver.max_batch` | 512 | most completions (frames) taken from the receive queue per poll |
| `scan.width` | 1000 | probe positions per row of the raster scan |
| `scan.height` | 1000 | rows of the raster scan |

## Operation

The application builds an HSDK 5 graph, compiles an execution plan, and runs it in a managed
session. The configured partition layout keeps Python messages in one process. SIGINT,
SIGTERM, the run-time limit, and handshake disconnection request cooperative session shutdown.
The existing CUDA events and stream waits govern CuPy array readiness and ring credits.
Wait for `[holoscan] ready` before starting acquisition. This marker follows GPU warm-up
and the first receiver activation; creating an asynchronous session alone does not mean
the pipeline is ready to receive at full rate.

A series is whatever arrives between its START and END frame. Its number of images is not known
in advance (`number_of_images` of the start message is only used as a size hint). The shape of its
scan is not announced by the sender either: it is the configured `scan` raster. A series whose
images are **not 96x96 uint16**, or an image frame outside a start/end pair, is rejected or
dropped with a message.

The END line reports `ring held at most N% (at image M)`: the largest share of the ring that was
read but not yet returned to imagemux, and how many images of the series had arrived at that
point. 100% means imagemux had to wait for ring space. If that happens near the start of a series,
look for a first-use cost; if it happens throughout, the pipeline is too slow for the rate.

## 4D-STEM output

The app puts a `ScanTensorOp` after `RdmaRxOp`, one per ring on that ring's GPU. Scans are always
raster: image id `i` of a series is at row `i // scan.width`, column `i % scan.width`.

The standalone graph uses typed `hs.pyobject` ports within one process. `RdmaRxOp` emits a
Python list of `Start`, `Batch` and `End` for each nonempty poll. `Batch.images` is a CuPy array
whose copy out of the ring may still be running; its CUDA event controls readiness and ring
credit. `ScanTensorOp` emits one `(index, frames)` pair on its `output` port:

| Item | Content |
|---|---|
| `frames` | dictionary, or `None` when there are no images: `diffraction` (k, 96, 96) uint16, `positions` (k, 2) int32 `[row, column]`, `ids` (k,) int64; CuPy arrays on the selected GPU |
| `index` | list in arrival order of `Start`, `Segment(series, start, stop)` (rows `start:stop` of `frames` belonging to that series), and `End` |

- **Per poll.** Nothing waits for a scan row or the end of the series. `receiver.max_batch`
  limits received transport frames per poll, and each transport frame can contain multiple
  images. The output image count is variable.
- **Complete when emitted.** `ScanTensorOp` synchronizes its stream before emitting, so consumers
  may read the arrays on any stream without waiting on CUDA events.
- **Lost and extra images.** Lost images leave their scan positions empty; images with IDs
  beyond `width * height` are left out with one message per series.
- **Paired data and metadata.** One message carries both `index` and `frames`, keeping series
  transitions paired with their images even when a poll spans more than one series.
- **PyTorch.** Not a dependency. A downstream operator can wrap the tensors without a copy with
  `torch.from_dlpack(frames["diffraction"])`.
- **Warm-up.** The dispatcher, `ScanTensorOp` and `VirtualDetectorOp` run one full batch
  (`receiver.max_batch` images) when they are initialised. That compiles the kernels and grows the
  memory pools before the first frame arrives, not during the first series. It makes startup a
  few seconds slower, mostly for CuPy compiling kernels in the fresh container. A custom consumer
  should warm up the same way.

`VirtualDetectorOp` (`examples/pipeline/operators.py`) is the example consumer. It integrates each
pattern over a bright-field disc and an annular dark-field ring, puts the two values at the image's
scan position in a (2, height, width) image, and prints the BF and ADF sums at the end of the series.


## RDMA

Short definitions of the acronyms are in the [Glossary](#glossary) at the end.

RDMA (Remote Direct Memory Access) lets the network card (NIC) of one machine write straight into
registered memory of another machine, without the receiver's CPU copying the data and without
a system call per message. The sender (i.e. DCU imagemux) produces frames at up to ~120 kHz (~18 Gbit/s);
with RDMA its NIC writes them into a buffer on the receiver (the *ring*) and the receiving CPU only has to
notice that a frame has arrived and, once consumed, tell imagemux that the space can be reused. This all means the GPU then
reads the directly images from the ring.

The transport is RoCE v2 (RDMA over Converged Ethernet) RDMA carried in UDP/IP packets over
ordinary Ethernet with an RC (Reliable Connected) queue pair: the network delivers every message
once, in order, or reports an error. All of it is driven through the `ibverbs` API (RDMA "verbs"),
here through `imxrdma`.

| Object | What it is here |
|---|---|
| **QP** (queue pair) | Releiable Connected (RC) connection endpoints: One on each side providing a send queue and a receive queue. Each has a number (QPN) and a state (`INIT` → `RTR` → `RTS`). |
| **CQ** (completion queue) | Destination for NIC to report finished work: one entry (completion, CQE/WC) per work request. |
| **MR** (memory region) | A block of memory registered with the NIC and visible to it. It has a local key (`lkey`) and a remote key (`rkey`). A peer may only access it with the right `rkey`, address range and access flags. |
| **PD** (protection domain) | Groups the QPs and MRs that belong together. `imxrdma.Context` owns one PD for its device. |
| **WR** (work request) | An operation posted to a queue (a write, or a receive buffer). Its id (`WRID`) comes back in the completion. |
| **Ring** | The MR that imagemux sequentially writes frames into; when the end is reached imagemux writes a *wrap* marker and continues at offset 0. |
| **Immediate** | 32 bits that travel with an RDMA write (WRITE_WITH_IMM). The receiver gets them in a completion, which is how we know that a frame arrived. |
| **Credit word** | An 8-byte value in *senders's* memory. The receiver RDMA-writes into it the ring position up to which it has finished reading; the sender never overwrites beyond it. |

Two kinds of operations are used:
- **RDMA WRITE (with immediate)**: sender → receiver, one-sided. The data goes into the ring at an
  address sender chooses, no receive buffer holds it. The *immediate* still consumes one posted
  RECEIVE on the receive queue and produces a completion, but that RECEIVE is zero-length.
- **RDMA WRITE**: sender → receiver, into the credit word.

### Dataflow:

```
 imagemux (sender)                                        pipeline (receiver)
 -----------------                                        ----------------------------
 SETUP (TCP, once)
                                  <------ connect ------  TCP client to senders's handshake port
                                  <------ Hello --------  receiver QP (qpn, psn, lid, mtu, gid) +
                                                          ring address / length / rkey
 Accept                           ----------------------> senders QP + credit word address / rkey
                                                          QP: INIT -> RTR -> RTS; post 1024
                                                          zero-length RECEIVEs
                                  <------ Ready --------  (TCP stays open as liveness signal)

 DATA (RDMA, per frame)
 WRITE header+data into the ring
 WRITE_WITH_IMM(last piece, imm = sequence) ------------> NIC places data in the ring
                                                          completion with the immediate -> poll
                                                          read header at the ring position, check
                                                          sequence, hand the images to the GPUs
 FLOW CONTROL
                                  <---- WRITE(position) - once the GPU kernels are done with the
 credit word := position                                  ring bytes (CUDA events)

 END: the TCP socket closes -> the receiver drains the ring for 1 s and stops.
```

1. **Handshake over TCP.** Receivers connect to senders handshake address (`<host>:<port>`, e.g. `10.42.41.10:31003`) and exchange
   three small messages: `Hello` (receivers: QP endpoint, ring address,
   length, `rkey`), `Accept` (senders QP endpoint, credit word address and `rkey`), `Ready`.
   All the numbers a QP needs to reach its peer (QPN, PSN, GID, MTU) are in these messages; there
   is no rdma_cm connection establishment.
2. **QPs.** `Context.create_qp` creates the QP (`INIT`); after `Accept`,
   `QueuePair.connect` moves it to `RTR` ("ready to receive", with senderss QPN, PSN and GID) and
   `RTS` ("ready to send"). 1024 zero-length RECEIVEs are posted, then `Ready` is sent.
3. **Data.** For each frame sender writes a 128-byte header and the image data at the next
   128-byte aligned ring position; the last write carries the sequence number as immediate. The
   RC guarantee (writes on one QP are applied in order, and the completion comes after the data)
   means that when receiver sees the completion the frame is in the ring. Receiver re-posts a RECEIVE for each
   one consumed.
4. **Reading.** `FrameReader` expects the frame at the current logical position, decodes the
   header, and checks sequence and position. Start/end frames delimit a series; image frames are
   handed to the GPU kernels with their ring addresses.
5. **Credit.** The ring is a circular buffer: senders may only write where receivers release space.
   Once the CUDA events show the kernels are done with the ring up to a position, receivers RDMA-write
   that position into the credit word. It is cumulative, so a lost or late credit costs nothing;
   The sender blocks when it would overtake the credit by more than the ring size.
6. **Failure.** A failed completion puts the QP in the error state and flushes every posted
   RECEIVE. `RdmaRxOp` prints "connection lost" with the first failed completion and the first
   flush (`QueuePair.errors`) and stops the app.

### Usage:

- **Frames.** Each frame is a 128-byte header (`FrameHeader`) followed by its data, at a
  128-byte aligned ring position. `kind` is START / END (Stream2 CBOR),
  image, or wrap (the writer skipped the rest of the ring). An image frame holds `batch_size`
  consecutive images of one channel; Only uncompressed 96x96 uint16 is supported (anything else raises).
- **Locating a frame.** The immediate is the frame's sequence number mod 2^32. Frames are in
  order, so `FrameReader` only tracks the logical position and checks the header it finds there
  (sequence and position); the image ids come from the header, so there is no wrap counting.
- **Receive credits.** Each WRITE_WITH_IMM consumes one pre-posted zero-length RECEIVE (no
  buffer). `RdmaRxOp` re-posts as many as it consumed, before it reads the frames. If the receive
  queue refuses some, the shortfall is carried over to the next tick (and the queue pair's
  statistics are printed once); more than 256 missing RECEIVEs is an error.
- **Flow control.** The credit word is a `u64` in the senders memory holding the consumed
  logical ring position (bytes). Once the GPU kernels that read a part of the ring are done
  (CUDA events, see "GPU processing"), `RdmaRxOp` RDMA-WRITEs the new position
  (`QueuePair.write_u64`, inline). It is cumulative, so a credit that cannot be posted (send queue
  full) is retried on the next tick.
- **Completion handling.** `QueuePair.poll` drains up to `max_batch` completions per call; a
  failed completion is counted in `n_failed`. Only the first failure and the first flush are
  described (`QueuePair.errors`), the rest counted, since a queue pair error flushes every posted
  RECEIVE.
- **Liveness.** `imagemux.py` (`Link`) keeps the handshake socket open (`Link.liveness`);
  when the sender closes it the app stops.
- **Announced rate.** The start line prints the rate the sender announces
  (`announced rate 120.0 kHz`, from `frame_time` of its start message) and the end line the number
  of images received; compare it with `number_of_images` and the DCU's `status/dropped`.

### Network: 

The receiver needs an interface with an IPv4 address on the same Ethernet segment as
the sender and an RDMA device on top of it (`rdma link show`). `connection.local_address` must use that address;
the RDMA device, port and GID index are found from it.

### Memory Regions:

There is a single memory region: the ring, 64 MiB (`connection.ring_mib`) of GPU memory (`connection.ring_gpu`) from
`cudaMalloc`, registered remote-writable (`dectris/connection.py`, `VerbsConn`). The RECEIVEs are
zero-length and the credit word is written inline, so neither needs a buffer.

Registering needs locked memory: `examples/pipeline/run.sh` starts the container with `--ulimit memlock=-1`,
`--privileged` and `--device /dev/infiniband` for that and for the RDMA devices.

## Multi-GPU

The receiver supports multiple GPUs. Each GPU gets its own **ring** buffer in that GPU's memory
that imagemux writes frames into. The imagemux decides which images go to which ring. For each ring the receiver keeps a `RingPath` (`operators.py`) with three parts: `FrameReader` which finds and decodes the frames imagemux has written into the ring, `Credits` which tell imagemux which parts of the ring it may overwrite again, `Dispatcher`, which launch the GPU kernels that copy the images out of the ring. The images of each ring are processed by their own `ScanTensorOp` and `VirtualDetectorOp`. There is only one ring currently pending imagemux support.

```
 ring 0 (GPU 0) --> copy kernel on GPU 0 --> ScanTensorOp 0 --> VirtualDetectorOp 0
 ring 1 (GPU 1) --> copy kernel on GPU 1 --> ScanTensorOp 1 --> VirtualDetectorOp 1
```

1. **Dispatch** (`dispatch.py`): Each received batch in a ring becomes one kernel on that
   ring's GPU with its own stream. The kernel copies the images straight from the ring to GPU
   memory; the ring space is credited back to imagemux once the copy is done. The START and END
   frames of a series are sent to every ring's operators.
2. **4D-STEM tensors** (`operators.py`): Each `ScanTensorOp` runs on its selected GPU and execution
   partition. It waits for each batch's event on its own stream, positions the images in the scan
   and emits them (see [4D-STEM output](#4d-stem-output)).
3. **Virtual detector** (`examples/pipeline/operators.py`): Each `VirtualDetectorOp` builds the
   BF / ADF images of a series from its ring's tensors. A series without an END frame is dropped
   when the next one starts. Nothing merges the rings yet: with several rings each detector only
   fills the scan positions of the images its ring received.

Today the handshake defines a single ring, so a run has one path. `connection.ring_memory` selects
GPU memory or mapped pinned host memory; `connection.ring_gpu` selects the GPU that processes it.

## Glossary

Terms and acronyms used in the code, configuration and logs.

**Network and RDMA**

| Term | Meaning |
|---|---|
| **ACS** | Access Control Services: a PCIe switch feature that can redirect peer-to-peer requests up to the root complex. Must be off between the NIC and the GPU for GPUDirect RDMA. |
| **CQ / CQE / WC** | Completion Queue / its entries (Completion Queue Entry, also called Work Completion): the NIC's report that a work request finished (or failed). |
| **DMA** | Direct Memory Access: a device reading or writing memory itself, without the CPU. |
| **GID** | Global Identifier: the 128-bit address of a RoCE port, derived from its IP (IPv4 `10.42.41.11` is `::ffff:10.42.41.11`). The GID *index* selects an entry of the port's GID table; v1 and v2 entries of one address are separate. |
| **GPUDirect RDMA** | The NIC reads/writes GPU memory directly over PCIe (needs `nvidia_peermem`). |
| **HCA** | Host Channel Adapter: the RDMA NIC (here a ConnectX-7). |
| **imm / immediate** | 32-bit value carried by RDMA WRITE_WITH_IMM; here the frame's sequence number mod 2^32. |
| **lkey / rkey** | Local / remote key of a memory region: what the owner and a remote peer must present to access it. |
| **LID** | Local Identifier: InfiniBand port address; 0 for RoCE (the GID is used instead). |
| **MR** | Memory Region: memory registered with the NIC. |
| **MTU** | Maximum Transmission Unit. The *path MTU* of a QP is an enum: 1 = 256 B ... 5 = 4096 B; the Ethernet MTU (9000 here) must be larger. |
| **NIC** | Network Interface Card (here the ConnectX-7; two ports, `roceP5p3s0f0/1`). |
| **PD** | Protection Domain: groups the QPs and MRs that may work together. |
| **PSN** | Packet Sequence Number: the starting packet number of each direction of an RC connection, exchanged at setup. |
| **QP / QPN** | Queue Pair / its number. |
| **RC** | Reliable Connected: the QP type that delivers each message once, in order. |
| **RDMA** | Remote Direct Memory Access. |
| **rdma_cm** | RDMA Connection Manager: library/kernel service that normally sets up connections; not used by imagemux (see "Queue pair without rdma_cm"). |
| **RoCE (v1/v2)** | RDMA over Converged Ethernet; v2 runs over UDP/IP and is routable. |
| **RQ / SQ** | Receive Queue / Send Queue of a QP. |
| **RTR / RTS / INIT** | QP states: Initialized; Ready To Receive (peer known); Ready To Send (fully connected). |
| **RX / TX** | Receive / transmit. |
| **SGE** | Scatter/Gather Entry: one (address, length, key) piece of a work request's buffer. |
| **verbs / ibverbs** | The Linux user-space API (`libibverbs`) for RDMA devices. |
| **WR / WRID** | Work Request / its identifier, returned in the completion. |
| **WRITE_WITH_IMM** | RDMA WRITE that also delivers a 32-bit immediate and consumes a posted RECEIVE at the target. |

