# Live4D Holoscan pipeline

One Holoscan SDK 5 application accepts either BSLZ4-compressed ZMQ input or uncompressed
imagemux RDMA input. Both routes support 96×96 uint16 images and feed the same raster scan
and bright-field/annular dark-field operators.

```text
ZMQ → pinned compressed batches → GPU BSLZ4 decode ─┐
                                                 ├→ DecodedBatch → scan positions → BF/ADF
RDMA → registered ingress ring → GPU tensor loan ─┘
```

## Project structure and algorithms

`live4d` is the application package under `holoscan_pipeline/python/`. It uses the reusable
`dectris` transport package and separates numerical algorithms from Holoscan scheduling.

| Source | Responsibility |
| --- | --- |
| `holoscan_pipeline/python/live4d/app.py` | Graph construction, transport selection, CLI, and application lifetime |
| `holoscan_pipeline/python/live4d/config.py` | Shared configuration and validation |
| `holoscan_pipeline/python/live4d/algorithms/virtual_detectors.py` | CuPy BF/ADF masks and per-frame integration |
| `holoscan_pipeline/python/live4d/processing.py` | Scan placement, ordered GPU processing, and observers |
| `holoscan_pipeline/python/live4d/completion.py` | Bounded event polling, completion metrics, and admission acknowledgements |
| `holoscan_pipeline/python/live4d/operators.py` | Python access to registered native ingest operators |
| `holoscan_pipeline/cpp/` | ZMQ reception, GPU decompression, RDMA gathering, and bounded publication |
| `holoscan_pipeline/schemas/` | Typed batch messages; CMake generates their Python and C++ bindings |
| `dectris/` | Imagemux protocol, TCP handshake, native verbs backend, and registered-ring ownership |

Add analysis methods such as ptychography or strain mapping under `live4d.algorithms` as
they are implemented. Algorithm modules accept diffraction frames and their numerical
parameters without importing Holoscan. Processing operators supply scan information,
schedule streaming or accumulated work, and manage GPU buffer lifetime.

## Build and run

Build from the repository root. Set the CUDA architecture for the target GPU (for example,
`121` for DGX Spark or `89` for Ada):

```bash
docker build -f docker/Dockerfile.holoscan \
  --build-arg CUDA_ARCHITECTURES=121 -t dectris-holoscan:hsdk5 .
```

The image includes Holoscan SDK 5.0.0, CUDA 13, nvCOMP, ZMQ, CuPy, and the reusable
`imx-receiver` native verbs package. Its entry point is `python3 -m live4d`.

Copy either `holoscan_pipeline/config/basic_ingest.yaml` or
`holoscan_pipeline/config/rdma.yaml` to `pipeline.yaml` and edit the transport settings.
The RDMA example uses documentation-only addresses; replace them with reachable addresses
on the sender and receiver RoCE interfaces.

```bash
docker run --rm --gpus all --network host \
  --ulimit memlock=-1:-1 --ulimit stack=33554432 \
  --device /dev/infiniband \
  -v "$PWD/pipeline.yaml:/etc/dectris.yaml:ro" \
  dectris-holoscan:hsdk5 --config /etc/dectris.yaml
```

For ZMQ, omit `--device /dev/infiniband` when the host has no verbs devices. Alternatively,
`docker compose -f docker/compose.holoscan.yaml up --build` runs the supplied ZMQ configuration.
The compose file grants verbs access so the same service can also run an RDMA configuration.

Wait for `[holoscan] ready` before starting acquisition. This marker follows receiver
initialization, GPU warm-up, and passage through the shared processing operators;
return from `run_async` alone does not establish readiness.

## Configuration and batching

`transport` selects `zmq` or `rdma`. The common settings are `batch_size`, `queue_depth`,
`device`, `scan.width/height`, `detector.bf_radius/adf_inner/adf_outer`, and
`execute.mode/seconds`. Unknown sections and keys are rejected. `batch_size` is between
1 and 1000; `queue_depth` is between 2 and 16 per graph edge.

The receiver publishes exactly `batch_size` images during an acquisition and flushes the
remainder before END. A batch never contains images from different series. Empty series
still publish ordered START and END messages. `execute.mode: finite` completes after the
first processed END; `service` continues across series. `seconds: 0` disables the time
limit. A positive limit begins at readiness. SIGINT/SIGTERM request cooperative shutdown;
shutdown does not fabricate an acquisition END or publish an incomplete acquisition.
RDMA sender disconnection retains the backend's one-second drain grace before stopping;
an abrupt disconnect does not guarantee completion of an acquisition still in flight.

Transport settings have separate meanings:

| Setting | Meaning |
| --- | --- |
| `batch_size` | Images in each downstream GPU batch, for either transport |
| `rdma.max_batch` | Maximum RDMA completions collected in one poll, from 1 to 1024 |
| Sender `rdma_batch_size` | Images packed into one transmitted RDMA frame; configured on the sender |
| `rdma.ring_mib` | Registered ingress ring capacity, independent of the tensor pool |
| `rdma.ring_memory` | `gpu` for GPUDirect RDMA or `host_mapped` for mapped pinned host memory |
| `zmq.endpoint` | Stream2 ZMQ PUSH endpoint consumed by the receiver's PULL socket |
| `rdma.remote_address` | Sender's imagemux TCP handshake address, including port |
| `rdma.local_address`, `rdma.device` | Receiver RoCE address and optional verbs device name |

For simulator RDMA batching above one, select uncompressed output (`compression=none`),
set bstream mode to `cbor-rdma`, and set `rdma_batch_size` explicitly after initialization.
ZMQ input requires `bslz4`. This application does not configure the sender.

## Shared messages and GPU ownership

`DecodedBatch` carries a monotonic `batch_id`, series ID, image IDs, and a runtime-owned
CUDA tensor. Separate START/END flags preserve series order. START retains its original
CBOR acquisition metadata, bounded to 16 KiB. END carries the received image count.
A readiness message is separate from every acquisition boundary.

`len(image_ids)` is the valid frame count. RDMA tensors reserve capacity for `batch_size`
images, including when END flushes a shorter batch; only the valid prefix may be read.
Control messages have no valid images. ZMQ decoded tensors may be sized to the valid count.
The graph queues and tensor bounds constrain in-flight storage. Application publication
windows limit outstanding messages to `queue_depth - 1` at each tensor boundary. One spare
queue slot covers input consumption at callback completion. Full windows pause ingress;
the RDMA source continues checking completed ring reads and returning eligible credits.
It retains at most one partially filled output tensor.

The RDMA ingress ring remains registered separately. The native source reads ring bytes
directly into an HSDK runtime tensor loan using the runtime CUDA stream. Write commits
record tensor readiness before publication. A cumulative ring credit becomes eligible only
when every preceding ring read has completed, including all pieces of a frame split across
output batches. Reads that fill a partial batch can complete and release ring credit before
that batch is full.

Scan placement uses each image ID: row `id // width`, column `id % width`. Images outside
the configured scan are counted as received but omitted from BF/ADF placement. The decoded
output feeds both operators through typed inputs. The scan operator sends position metadata;
the detector pairs it with the original batch using its event time and checks the batch ID.
GPU tensors remain visible to runtime capacity and readiness tracking. CuPy views are created
on the callback's runtime stream. The SDK protects storage through producer readiness,
each consumer's queued CUDA work, and retained Tensor/CuPy views, even after `compute()` returns.
Tensor recycling does not depend on an application acknowledgement.
BF/ADF values retain float32 processing. Reported totals use a
float64 reduction of those stored values to avoid large-scan float32 reduction rounding.

### Contract for additional algorithms

The supported runtime is Holoscan SDK 5.0.0. Its
[Tensor stream and pool contract](https://docs.nvidia.com/holoscan/sdk-user-guide/data-and-payloads/tensors-in-depth)
and [Python Tensor API](https://docs.nvidia.com/holoscan/sdk-user-guide/python/api-reference/holoscan)
define the ownership rules used here:

- Import the received runtime Tensor with `cp.from_dlpack(batch.frames)` inside
  `with ctx.stream():`. CuPy must use that runtime stream, as `VirtualDetectorOp` does
  with `cp.cuda.Stream.from_external`. Producer readiness is already ordered there.
  Submit all work and record its completion event before the callback returns. Borrowed
  SDK stream handles must never be saved for submissions from later callbacks or threads.
- Retain the Tensor or its CuPy view when later work needs the input. Reimport the
  retained Tensor on a later callback's current stream before using it there. Release
  retained views in the operator that last used them, or establish completion before
  releasing them elsewhere. Independent consumers each import on their own callback
  stream; an acknowledgement from one consumer cannot retire another consumer's storage.
- Use `CompletionTracker.wait_for_previous(stream)` when an algorithm mutates state
  across callbacks. It inserts a GPU event dependency, allowing callbacks to return
  while preserving series and scan-write order. Algorithms themselves remain in
  `live4d.algorithms` and accept arrays and numerical parameters without Holoscan imports.
- Call `CompletionTracker.track()` inside the callback after the algorithm and observer
  submit work. Retain algorithm scratch through its `owners` argument. Its polling
  thread only queries recorded events; it makes no submissions on borrowed streams.
  Drain and close the tracker before destroying native windows or transport resources.
- Register every consumer in `_PublicationWindow(..., consumers=(...))`. Give each
  consumer its own acknowledgement callback. Acknowledgements identify exact batch IDs;
  out-of-order completion cannot skip earlier batches. Admission advances only after
  every registered consumer completes each batch. A metadata-only consumer may acknowledge
  after publishing its metadata; GPU consumers acknowledge after their event completes.
  The existing graph registers scan and detector separately.
- Keep runtime-owned output loans for inter-operator image data and commit writes on
  the producing callback's stream before publication. Bind the output device explicitly.
  New output boundaries need their own finite tensor bounds and admission budget.
  Runtime pool exhaustion fails explicitly; retained views consume that finite pool.
  CuPy pass-through outputs require a separate bound on allocations and submitted work.

`CompletionTracker` has a fixed pending-event capacity matching the publication window.
It polls independently of detector callbacks, so a one-batch window can make progress.
It retains owners until completion, propagates asynchronous failures, and drains outstanding
events during shutdown. Exhaustion raises an error rather than growing storage or waiting
for a routine batch. Its acknowledgements control application admission, while the SDK
independently controls tensor reuse and the RDMA source independently returns ring credits.

Observers run on the current callback's stream. `positioned.images` is an asynchronous
CuPy view of the valid input prefix, and `positioned.tensor` is its retained SDK Tensor.
Keeping either retains storage and can exhaust the finite pool. To use retained input on
a later callback's stream, reimport `positioned.tensor`. An observer that needs host-ready
input can opt into `PipelineApp(..., observer_host_ready=True)`; explicit `cp.asnumpy` in
an observer is also a blocking readback. Enqueued observer work is included in completion
and admission tracking. END observers receive host-ready totals.

Blocking waits remain at startup warm-up, requested observer readback, END aggregation,
validation, error cleanup, and teardown. The ZMQ decoder also reads GPU block counts and
nvCOMP validation statuses on the host. These waits do not pace normal detector batches.
The [asynchronous consumer tests](../holoscan_pipeline/tests/test_rdma_batches.py)
enqueue delayed GPU reads, release callback-local views, and verify pixels after execution
has drained. A separate pressure test disables application admission, fills the finite
SDK pool, and checks that storage used by pending reads is preserved until explicit
allocation exhaustion. Retained inputs and fanout are also covered.

## Validation and measurements

Run focused tests in the built image, mounting the repository tests:

```bash
docker run --rm --gpus all --network host --device /dev/infiniband \
  --ulimit memlock=-1:-1 -v "$PWD:/work:ro" \
  --entrypoint python3 dectris-holoscan:hsdk5 \
  -m pytest -q -p no:cacheprovider /work/holoscan_pipeline/tests
```

RDMA loopback tests require `IMXRDMA_LOCAL_ADDR` set to an active receiver RoCE address and
passed into the container with `-e IMXRDMA_LOCAL_ADDR`; also pass `-e RDMA_TEST_IN_IMAGE=1`
to enable them in the HSDK image. Tests cover fixed batching, partial
and empty acquisitions, consecutive series, pixel values, scan clipping, BF/ADF results,
readiness, credits, and shutdown. Protocol tests remain under `dectris/tests`; five existing
probe tests currently reference the absent `tools.probe_imagemux` helper.
GPU-ring cases also require verbs registration of CUDA device memory. A registration
failure that reports missing `nvidia_peermem` and `Bad address` skips those cases;
mapped-host tests still exercise CUDA consumers, asynchronous lifetime, and shutdown.

At each completed series, `[series-result]` reports received/placed images, continuity gaps,
batch sizes, BF/ADF totals, and delivered throughput. `delivered_fps` measures images between
the first completed batch and the last completed image batch, excluding the first batch
from its numerator. It includes shared scan/BF/ADF processing. Compare it with separately
recorded sender image counts, source acquisition duration, bstream drop deltas, and NIC
counters. NIC sampling measures transport traffic and includes protocol overhead; it is not
a substitute for delivered image counts.
Completion timestamps come from ordered event polling, with approximately millisecond
polling resolution; this is a completion-based rate, not a CPU submission rate.

For sample pixel validation, `python3 -m live4d.validate` accepts an NPZ containing
`ids` and CPU-reference `frames`, an expected image count, and a ZMQ endpoint. It exercises
the same downstream processing and reports mismatched samples as failures.
