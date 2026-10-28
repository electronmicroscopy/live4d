# Live4D Holoscan pipeline

One Holoscan SDK 5 application accepts either BSLZ4-compressed ZMQ input or uncompressed
imagemux RDMA input. Both routes support ARINA 96×96 (2×2 binned) and 192×192 (unbinned, 1×1)
uint16 images and feed the same raster scan and bright-field/annular dark-field operators.
Each series' START metadata selects the image dimensions automatically. Service mode can
process consecutive series with different dimensions. Image dimensions and type must match
the series metadata.

```text
ZMQ → pinned compressed batches → GPU BSLZ4 decode ─┐
                                                 ├→ DecodedBatch → scan positions → BF/ADF
RDMA → registered ingress ring → GPU tensor loan ─┘              └→ optional NXmx HDF5 writer
```

## Project structure and algorithms

`live4d` is the application package under `holoscan_pipeline/python/`. It uses the reusable
`dectris` transport package and separates numerical algorithms from Holoscan scheduling.

| Source | Responsibility |
| --- | --- |
| `holoscan_pipeline/python/live4d/app.py` | Graph construction, transport selection, CLI, and application lifetime |
| `holoscan_pipeline/python/live4d/config.py` | Shared configuration and validation |
| `holoscan_pipeline/python/live4d/algorithms/virtual_detectors.py` | CuPy BF/ADF masks and per-frame integration |
| `holoscan_pipeline/python/live4d/processing.py` | Scan placement and Holoscan processing operators, including CUDA synchronization and tensor lifetime |
| `holoscan_pipeline/python/live4d/operators.py` | Python access to registered native ingest operators |
| `holoscan_pipeline/cpp/` | ZMQ reception, GPU decompression, RDMA gathering, bounded publication, and NXmx HDF5 writing |
| `holoscan_pipeline/schemas/` | Typed batch messages; CMake generates their Python and C++ bindings |
| `dectris/` | Imagemux protocol, TCP handshake, native verbs backend, and registered-ring ownership |

Add analysis methods such as ptychography or strain mapping under `live4d.algorithms` as
they are implemented. Algorithm modules accept diffraction frames and their numerical
parameters without importing Holoscan. Processing operators supply scan information,
schedule streaming or accumulated work, and manage GPU buffer lifetime.

## Build and run

The commands below use Bash on a Linux GPU host with Docker, NVIDIA GPU container access,
and a driver compatible with the image's CUDA 13 runtime. RDMA also requires an active
verbs/RoCE interface and working memory registration for the selected ring type. Installing
the Python protocol package alone does not provide those host capabilities.

Build from the repository root. Set the CUDA architecture for the target GPU (for example,
`121` for DGX Spark or `89` for Ada):

```bash
docker build -f docker/Dockerfile.holoscan \
  --build-arg CUDA_ARCHITECTURES=121 -t dectris-holoscan:hsdk5 .
```

The image includes Holoscan SDK 5.0.0, CUDA 13, nvCOMP, ZMQ, CuPy, HDF5 1.10 with h5py and
hdf5plugin, and the reusable `imx-receiver` native verbs package. Its entry point is `python3 -m live4d`.

Copy either `holoscan_pipeline/config/basic_ingest.yaml` or
`holoscan_pipeline/config/rdma.yaml` to `pipeline.yaml` and edit the transport settings.
The RDMA example uses documentation-only addresses; replace them with reachable addresses
on the sender and receiver RoCE interfaces.

`holoscan_pipeline/config/rdma_rings.yaml` is the protocol 3 example: four 64 MiB rings for an
imagemux that distributes series over several receiver rings. See
[Multiple RDMA rings](#multiple-rdma-rings).

`holoscan_pipeline/config/buffle_rdma.yaml` configures the live Spark-to-Buffle link:
sender `10.10.10.2:31003` on Spark's `enp1s0f0np0`, receiver `10.10.10.1` on Buffle's
`enp185s0f0np0` / `mlx5_0`, a 64 MiB mapped-host ring, and a 1000×1000 raster in service
mode. CUDA device 0 is relative to the GPUs exposed to the container. Mount this YAML
as the application config to use that deployment.

```bash
# Choose one starting configuration:
cp holoscan_pipeline/config/basic_ingest.yaml pipeline.yaml
# For RDMA, use holoscan_pipeline/config/rdma.yaml instead.
```

For ZMQ, replace `tcp://127.0.0.1:5555` with the sender's Stream2 endpoint unless the sender
actually runs on this host and port. For RDMA, start the sender's handshake service before
launching the receiver, but wait for receiver readiness before triggering images.

```bash
docker run --rm --gpus all --network host \
  --ulimit memlock=-1:-1 --ulimit stack=33554432 \
  --device /dev/infiniband \
  -v "$PWD/pipeline.yaml:/etc/dectris.yaml:ro" \
  dectris-holoscan:hsdk5 --config /etc/dectris.yaml
```

For ZMQ, omit `--device /dev/infiniband` when the host has no verbs devices.

Wait for `[holoscan] ready` before starting acquisition. This marker follows receiver
initialization, GPU warm-up, and passage through the shared processing operators;
return from `run_async` alone does not establish readiness.

### Protocol 3 on Buffle with two GPUs

`holoscan_pipeline/config/buffle_rdma_lanes.yaml` runs the Spark-to-Buffle link with protocol 3
and two GPU lanes. The HDF5 writer is off in this file; set `writer.enabled: true` to write
files. Work through these steps in order.

1. Ask the sender's owner which imagemux protocol its software speaks, and set `rdma.protocol`
   to match. A mismatch fails with imagemux closing the connection after the Hello.
2. The sender must send RDMA (`cbor-rdma`) from the interface in `rdma.remote_address`, with
   uncompressed images. With `devices` and protocol 3 the images are not decompressed.
3. Copy the YAML and set `scan` to the raster of the acquisition. The raster comes from this
   file, not from the sender, unless the series' `header_appendix` carries one.
4. Create the config file and the writer's output directory before `docker run`. Docker creates
   a missing mount source as a root-owned directory, and the receiver then fails with
   `IsADirectoryError: '/etc/dectris.yaml'`.
5. Expose the GPUs to the container with `--gpus '"device=1,3"'`. `devices: [0, 1]` then selects
   the first and second of those, whatever their numbers on the host.

```bash
docker run --rm --gpus '"device=1,3"' --network host \
  --ulimit memlock=-1:-1 --ulimit stack=33554432 \
  --device /dev/infiniband \
  -v "$PWD/data:/data" \
  -v "$PWD/buffle_rdma_lanes.yaml:/etc/dectris.yaml:ro" \
  dectris-holoscan:hsdk5 --config /etc/dectris.yaml
```

6. Wait for `[holoscan] ready`, then trigger. Series go to the lanes in turn, and the `device`
   field of each `[series-result]` shows the lane's device.
7. In service mode the receiver exits when the sender closes its connection, for example when
   the sender is restarted or switches streams. Start it again before the next acquisition.
   `execute.mode: finite` stops after the first acquisition.
8. With `writer.enabled: true` the writer needs disk space: 96×96 images are about 18 GB
   uncompressed per million images, and the files are smaller by a data-dependent ratio. A series without an
   `electron_microscope` map in its `header_appendix` gets no `_em_metadata.h5` from the writer.

### Docker Compose alternative

`docker compose -f docker/compose.holoscan.yaml up --build` uses the ZMQ YAML copied into
the image. The supplied service sets CUDA architectures to `86;89`, requires
`/dev/infiniband` even for ZMQ, and does not mount the root-level `pipeline.yaml` above.
Before using Compose, edit its build architecture for your GPU, remove the `devices`
entry on a host without verbs devices, and either edit the shipped YAML and rebuild or
mount your configuration and change `command` to its container path. Selecting RDMA
requires that configuration change; granting verbs access alone does not select it.

### CLI overrides and development builds

`--transport`, `--endpoint` (ZMQ only), `--batch-size`, `--queue-depth`, `--device`, and `--devices`
override their YAML values. `--plan /path/plan.json` writes the compiled graph plan and
then runs the application; it is not a dry run. RDMA performs its handshake after GPU warm-up.
Use a writable bind mount to retain a plan outside the container.

The image runs the Python package, native library, and generated schema bindings from
`build/holoscan_pipeline/python`. Mounting the repository at `/work` only changes the
files under `/work`; it does not replace the installed application. Rebuild the image
after source or schema edits before testing those changes. `python3 -m live4d` without
`--config` uses the defaults in `live4d/config.py`; the image's default CMD supplies the
shipped ZMQ YAML.

## Configuration and batching

`transport` selects `zmq` or `rdma`. The common settings are `batch_size`, `queue_depth`,
`device`, `scan.width/height`, `detector.bf_radius/adf_inner/adf_outer`,
`execute.mode/seconds`, and `writer`. Unknown sections and keys are rejected. `batch_size` is between
1 and 1000; `queue_depth` is between 2 and 16 per graph edge.

`devices: [0, 1]` selects an ordered list of visible CUDA ordinals for either transport.
It overrides the legacy scalar `device`. Omitting it uses `[device]`; explicitly setting
`devices: [0]` follows exactly the same path as longer lists. `--devices 0 1` overrides the list;
`--device 0` clears it and selects one device. Device numbers are relative to the
container's `CUDA_VISIBLE_DEVICES` selection.

Each list entry creates an independent acquisition-routing lane. Repeated entries such
as `devices: [0, 0]` run two branches on one GPU for development and testing, with separate
buffers, result slots, and admission windows (and registered rings for RDMA). Acquisitions
alternate between the entries and then reuse the first branch. They share the physical
GPU's compute and memory capacity; this does not test transfers or isolation between GPUs.

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
| `rdma.ring_mib` | Registered capacity of each ingress ring, independent of the tensor pool |
| `rdma.ring_memory` | `gpu` for GPUDirect RDMA or `host_mapped` for mapped pinned host memory |
| `rdma.protocol` | imagemux RDMA protocol: `2` (default, one ring) or `3` (one or more rings) |
| `rdma.rings` | Protocol 3 ring count, from 1 to 256; default 1. Protocol 2 requires 1 |
| `zmq.endpoint` | Stream2 ZMQ PUSH endpoint consumed by the receiver's PULL socket |
| `rdma.remote_address` | Sender's imagemux TCP handshake address, including port |
| `rdma.local_address`, `rdma.device` | Receiver RoCE address and optional verbs device name |
| `rdma.timeout` | TCP handshake timeout in seconds, positive; default 10 |

Default downstream settings are 512 images/batch, queue depth 4, CUDA device 0, and a
1000×1000 raster. When the START `header_appendix` map contains integer `scan_width` and
`scan_height`, that series uses them instead of `scan.width/height`; otherwise it uses the
configured raster. A START that announces only one of them, or a non-positive or non-integer
value, is rejected. All START `header_appendix` keys are listed in
[header_appendix keys](header_appendix.md). The configured raster also sizes the READY warm-up. The RDMA example selects `host_mapped`, while an
omitted `rdma.ring_memory` defaults to `gpu`. An omitted `detector` section uses BF radius
16 and ADF inner/outer radii 24/48.

Detector radii are in input-image pixels about the image center: `(47.5, 47.5)` for
96×96 and `(95.5, 95.5)` for 192×192. BF includes pixels at or inside its radius; ADF
includes both annular boundaries. Radii must satisfy `0 <= bf_radius <= 136` and
`0 <= adf_inner <= adf_outer <= 136`. Defaults remain 16/24/48 for both sizes;
to cover the same detector area after switching from 96×96 to 192×192, double the
radii (for example, 32/48/96). Masks for both sizes are built at startup and selected
for each series. Editing YAML while running does not change the radii.

For simulator RDMA batching above one, select uncompressed output (`compression=none`),
set bstream mode to `cbor-rdma`, and set `rdma_batch_size` explicitly after initialization.
ZMQ input requires `bslz4`. This application does not configure the sender.

A decoded uint16 frame occupies 18,432 bytes at 96×96 or 73,728 bytes at 192×192;
a full default batch of 512 images occupies 9 MiB or 36 MiB, respectively. HSDK decoded
tensor pools reserve 70.3 MiB per slot for the maximum supported 1000×192×192 uint16
batch. The compiled graph determines the slot count. Smaller tensors use the same slots;
switching to binned images does not shrink the pool allocation.

The two float32 scan maps use `8 * width * height` bytes of the series' raster, before temporary
arrays, retained prior-series results, runtime tensors, and decoder scratch. The RDMA
ring is additional storage. Scan size, batch size, and queue depth affect memory demand.

## Multiple RDMA rings

imagemux protocol 3 lets the receiver register several rings. imagemux writes each series into
one ring and moves to the next ring, round robin, when the series ID changes; the first series
of a connection goes to ring 0. Each ring has its own positions, sequence numbers, and credit
word. While the receiver still holds one series' ring bytes, imagemux can write the next series
into the next ring. Protocol 2 imagemux builds accept only `rdma.protocol: 2`, and protocol 3
builds accept only `rdma.protocol: 3`, including with one ring.

`rdma.protocol: 3` selects the `RdmaRingsRxOp` source instead of `RdmaRxOp`. The receiver
allocates `rdma.rings * rdma.ring_mib` MiB as one registered GPU or mapped-host region and
advertises ring i at offset `i * ring_mib` MiB. Every ring must hold the largest frame: a frame
larger than its ring is dropped by imagemux. Downstream messages, batching, readiness, and
tensor ownership are the same as with protocol 2.

The source processes one series at a time, in the sender's order. It reads only the ring of the
active series and queues the completions announced for other rings until END advances it to the
next ring. Credits are returned per ring with the same GPU-read ordering as protocol 2. Receive
completions arrive in write order. If the active ring has no queued frames while another ring
has frames, the sender dropped the rest of the active series, for example its END after a ring
stayed full for longer than imagemux's send timeout. The source then fails with
`Invalid RDMA stream: ring N received a frame before ...` instead of waiting indefinitely.
An immediate that names a ring beyond `rdma.rings`, or a header whose ring, sequence, or
position does not match, also fails the run.

### Protocol 3 with `devices`

With `rdma.protocol: 3` and `devices` set, the receiver keeps one connection and reads its
rings on the CPU, so `rdma.ring_memory` must be `host_mapped`. `RdmaRingsHostRxOp` copies each
frame into pinned host staging, returns the ring credit as soon as the copy completes, and tags
every batch with the lane that processes its series: series k goes to lane k % len(devices),
whatever ring the sender wrote it into. One `RawUploadGpuOp` per lane copies that lane's batches
to its GPU, where the scan, detector, and HDF5 writer operators run as with other lanes. The
sender must send uncompressed images. A series starts only when its lane has finished the
previous one. Without `devices`, protocol 3 uses `RdmaRingsRxOp` as described above. The ANS
encoder is not supported with `devices`.

## HDF5 output

`writer.enabled: true` adds `H5WriterOp` as a second consumer of decoded batches. Each series
is written as `<writer.output_dir>/<name>_master.h5` plus `<name>_data_NNNNNN.h5` data files,
where `<name>` is `writer.name_pattern` (default `series_$id`) with `$id` replaced by the series ID.
A non-empty string `name_pattern` in the START `header_appendix` overrides the configured pattern
for that series; a missing or empty one keeps it. Either pattern must stay below `output_dir`:
absolute paths, `..` components, and a trailing `/` or `.` are rejected, and a rejected or
non-string appendix pattern fails the run. Subdirectories in the pattern are created. Existing
files with the same names are overwritten. The `[h5-series]` record reports `name_source` as
`header_appendix` or `config`.

A boolean `enable_h5_writer` in the START `header_appendix` turns writing off for one series when
`false`; `true` or a missing key writes the series. A disabled series creates no files, and its
images skip GPU compression; its `[h5-series]` record has `"enabled": false`. A non-boolean value
fails the run. The key cannot enable writing when `writer.enabled` is false, because the writer is
then not part of the graph.
The layout follows the imagemux filewriter's NXmx 2024 format: the master file holds START
metadata (beam, detector, per-channel threshold energy, flatfield and pixel mask, detector
translation, and goniometer axes when announced) and a virtual `/entry/data/data` of shape
`(number_of_images, 1, height, width)` over the data files. Each data file holds up to
`writer.images_per_file` images (default 1,000,000) in `/entry/data/data` with one chunk per
image. Files appear under their final names when complete; `.tmp` names mark files in progress.
Per-image `start_time` datasets are not written because batches carry no image timestamps.
Unlike imagemux, which uses HDF5 1.14 format bounds, files use the HDF5 1.10 format.

`writer.format: em` selects imagemux's DECTRIS EM format (`DECTRIS_EM`, novena's `hdf5 em 1.0`):
the same files, plus an external link `/entry/instrument/electron_microscope` in the master file to
`/electron_microscope` in `<name>_em_metadata.h5` beside it. The link stores only the file name, so
the file set can move. When the START `header_appendix` contains an `electron_microscope` map,
the writer creates the metadata file from it at START, in novena's layout; otherwise the
acquisition controller writes the file, and the link resolves once it exists. The `[h5-series]`
record reports `em_metadata_file`, or `null` when the writer did not create one. The default
`nxmx` writes neither. See [header_appendix keys](header_appendix.md#em-metadata-electron_microscope).

```yaml
writer:
  enabled: true
  output_dir: /data          # mount a writable host directory here
  name_pattern: series_$id
  format: nxmx               # em adds the link to <name>_em_metadata.h5
  images_per_file: 1000000
  staging_batches: 4         # pinned staging batches between GPU compression and disk
```

Chunks use the bitshuffle HDF5 filter (ID 32008) with zstd, filter options `(0, 5, 2, 0, 3)` as in
imagemux, so any reader with the standard bitshuffle plugin decodes them, for example h5py with
`import hdf5plugin`. The GPU bit-shuffles each image in 4096-element blocks, nvCOMP compresses
the blocks with zstd (at its fixed default level), and a framing kernel writes the bitshuffle
chunk headers into mapped pinned staging. One writer thread performs every HDF5 call and stores
the chunks with `H5Dwrite_chunk`, without filtering on the host. With `devices`, each lane has its
own writer; their HDF5 calls take one process-wide lock, so lanes write one at a time.

The data layout uses START `number_of_images`, which the writer therefore requires. Images whose
IDs fall outside it are counted as `dropped_images` and not written. Unwritten positions read as
the fill value 65535. A series that announces zero images writes no files. A full staging ring
delays the writer's window release, which pauses ingress like a slow detector; it does not drop
images. END waits until the series' files are closed, and a write failure fails the run. In
finite mode the later of the detector and writer requests completion, because requested
completion does not drain queues. At END the writer prints an `[h5-series]` JSON record with
written and dropped counts, data files, compressed bytes, and `written_fps` over the series'
write interval.

## ANS encoding

`encoder.enabled: true` adds `AnsEncoderOp` (`processing.py`) as one more decoded-batch consumer:
window consumer 1 without the HDF5 writer, 2 with it. It runs in its own `encoder` partition and
keeps every series on the GPU in lossless ANS form through quantem.gpu's `LiveEncoder`, which also
corrects flagged pixels. The result of each series is the `Dataset4dstemGPU` that
`quantem.gpu.io.load` returns for the same counts. Build `docker/Dockerfile.ans` on the pipeline
image to install quantem.gpu.

```yaml
encoder:
  enabled: true
  keep: 1        # series kept resident on the GPU by the default sink (0: all)
```

Each activation copies the batch's images once, on the runtime stream, into a reused staging chunk
of 4096 patterns (`algorithms/ans_encoding.py`) and releases the batch without a host wait. A full
chunk is encoded in place; `LiveEncoder` reads the encoded sizes back once per chunk, which blocks
only the encoder partition. The scan raster of a series is the `header_appendix` scan, else a square
raster of `number_of_images`, else `scan`. A series whose image ids skip or repeat is dropped by the
encoder; the detector still reports it. In finite mode the run completes after the last END
consumer, encoder included. Each kept or dropped series prints an `[ans-result]` record with
`images`, `encoded_bytes`, `raw_bytes`, `kept` and `encode_interval_s`.

`PipelineApp(config, encoder_sink=...)` replaces the default sink (`ResidentSeries`), for example
to save each series or to show it while it arrives; the sink contract is in `ans_encoding.py`.

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
the series' scan are counted as received but omitted from BF/ADF placement. The decoded
output feeds both operators through typed inputs. The scan operator sends position metadata;
the detector pairs it with the original batch using its event time and checks the batch ID.
GPU tensors remain visible to runtime capacity and readiness tracking. CuPy views are created
and consumed inside the typed detector callback's runtime stream scope. HSDK5 orders input
readiness and release on that stream, so callbacks can return with GPU work pending while
tensor storage remains protected from premature reuse. No separate owner queue is needed.
See the SDK's [tensor ownership and stream contract](https://docs.nvidia.com/holoscan/sdk-user-guide/data-and-payloads/tensors-in-depth).
Mask initialization and READY warm-up retain host waits. END queues the totals copy on
the runtime stream and records a CUDA event; the application polls completion without
blocking other GPUs. Both transports use the same asynchronous result infrastructure. One result
slot per routing lane retains the map, totals, and END callback until the event completes. It never
retains the runtime stream. Results are published only after their copy completes.
BF/ADF values retain float32 processing. Reported totals use a
float64 reduction of those stored values to avoid large-scan float32 reduction rounding.

Maps are cleared at START. Unacquired positions remain zero, so a zero alone does not
distinguish missing data from measured zero intensity. `gap_events` reports discontinuities
in the received ID sequence; `missing_images` counts forward skips, not an unreceived tail
after the final ID. Compare `images` with the sender's count and `announced_images` as well.
A successful END validates the receiver's internal count but does not require it to equal
the announced count. Maps stay in GPU memory; the CLI reports totals, not image files.

## RDMA acquisition routing

This section describes `rdma.protocol: 2`; protocol 3 with `devices` is described under
[Multiple RDMA rings](#multiple-rdma-rings).

For round-robin acquisition routing, select `transport: rdma`, `execute.mode: service`,
and `devices: [0, 1]` (or a longer ordered list). Whole acquisitions stay on one GPU;
subsequent acquisitions advance through the list. Finite mode still processes just one
acquisition, on the first device. Numerical kernels, scan placement, and detector
processing are shared with the single-device graph and ZMQ.

The application creates one fixed-device source/scan/detector branch and one registered
ring per list entry, then compiles and runs one graph. Only one sender connection is active.
After END is published, the source drains ring reads and credit-send completions. The
controller requests detach, waits for its generation-tagged acknowledgment, closes the
old connection, and handshakes using the next ring. Sources are never restarted between
acquisitions. Ring addresses and rkeys persist; QPs, wire sequence numbers, and credits
start fresh per connection. Published batch IDs remain monotonic within each branch.

Ingress drain is independent of BF/ADF completion because the source has copied the ring
into runtime-owned tensors. The next GPU can process while the previous GPU finishes.
Before reconnecting a lane, the controller waits for its preceding result; this bounds
pending acquisitions to one per lane. A list containing one GPU uses the same lifecycle,
including reconnection after every acquisition in service mode.

**Wait for `[rdma-ready]` before each ARM**, including with one GPU: START can be sent
during arming. This notification identifies the zero-based acquisition index, CUDA
device, and connection generation. Sender connection status or a previous series result
alone does not prove the next receiver is ready. `[holoscan] ready` and `PipelineApp.ready`
are one-time startup notifications. Embedded acquisition controllers can pass
`receiver_ready=callback` to `PipelineApp`; the callback receives the same dictionary
on the application's control thread and should enqueue work without blocking. Keep
detector ARM/TRIGGER/DISARM sequencing in the acquisition controller; Live4D does not
configure or trigger the detector. Uncoordinated back-to-back acquisitions on the old
connection are rejected rather than silently routed to the wrong GPU.

`[series-result]` includes the CUDA `device`. Different GPUs may complete out of order.
The optional data observer still runs inside the detector activation's runtime stream.
For both transports, its END notification runs on the control thread after CUDA completion; treat
that notification as completed metadata and do not submit work using its input tensor.
On shutdown or failure the runtime joins before QPs, pending results, mailboxes, and
registered rings are released. A `PipelineApp` is for one run.

## ZMQ acquisition routing

ZMQ uses the same `devices` selection and whole-acquisition round-robin order. One PULL
socket receives the ordered START/images/END stream. The source publishes each compressed
batch once in a pinned runtime tensor, with a destination index. Fixed-device decoders
share that read-only host tensor through graph fanout; only the selected decoder allocates
GPU output and performs H2D/decompression. Each GPU then uses the same scan and detector
operators as RDMA. No decoded tensor or numerical operation crosses devices, and the
single-device case follows the same path.

The supported image envelope is `data.threshold_1`, tensor tag 40, uint16 typed-array tag
69, and BSLZ4 compression tag 56500. The native parser follows that structure and skips
unrelated metadata; tag-like bytes inside image IDs or metadata are not tensor tags.

One compressed publication window per decoder bounds fanout queue occupancy; inactive
decoders release only their own input credit. Decoded batch IDs are monotonic within each
routing branch. A separate acquisition window per list entry admits a new START only after
that branch's preceding result completes. Other branches can process while a prior result is pending.
When the destination is busy, ingress pauses before reading START and ZMQ/TCP applies
backpressure. Runtime tensor pools, graph queues, and result slots remain bounded. The
fanout means a slow decoder can also delay delivery to other branches; there is no
unbounded intermediate acquisition queue. Sender drop behavior under backpressure still
depends on the detector's ZMQ buffering policy.

Wait for `[holoscan] ready` before the first ARM. ZMQ keeps its socket across acquisitions
and routes subsequent START messages automatically, so it needs no RDMA-style per-ARM
reconnection notification. START can arrive during ARM, and consecutive series may change
geometry. Results include the CUDA device and may complete out of acquisition order.
Finite mode admits only the first acquisition, on the first configured GPU; queued later
START messages remain unread. The control thread requests cooperative stop only after
that acquisition's CUDA event and observer callback complete. The sample
validator also requires one completed acquisition, exact counts, and checked reference
pixels before accepting that termination.

## Validation and measurements

### Configuration checks without GPU libraries

From the repository root, in a Python 3.12+ environment:

```bash
python3 -m pip install pytest pyyaml
PYTHONPATH=holoscan_pipeline/python python3 -m pytest -q \
  holoscan_pipeline/tests/test_pipeline_settings.py
```

See the [transport package tests](../dectris/README.md#tests) for CPU protocol checks.

### GPU and RDMA integration checks

Rebuild the image from the source being tested, then mount the repository tests:

```bash
docker run --rm --gpus all --network host --device /dev/infiniband \
  --ulimit memlock=-1:-1 -v "$PWD:/work:ro" \
  --entrypoint python3 dectris-holoscan:hsdk5 \
  -B -m pytest -q -p no:cacheprovider /work/holoscan_pipeline/tests
```

RDMA loopback tests require `IMXRDMA_LOCAL_ADDR` set to an active receiver RoCE address and
passed into the container with `-e IMXRDMA_LOCAL_ADDR`; also pass `-e RDMA_TEST_IN_IMAGE=1`
to enable them in the HSDK image. Tests cover fixed batching, partial
and empty acquisitions, consecutive series at both image sizes, size changes between
series, geometry rejection, pixel values, scan clipping, BF/ADF results, readiness,
credits, shutdown, and HDF5 output. The HDF5 tests read the files back with h5py and
hdf5plugin in a separate interpreter, because h5py bundles its own HDF5 library. For ZMQ-only checks, omit the verbs device argument;
RDMA tests will skip without their two opt-in variables. The ZMQ tests generate their own
messages and do not require a detector or reference dataset. A green run with skipped RDMA
tests is not RDMA acceptance. These application tests require the built image and GPU;
running the whole suite on a CPU-only host does not automatically skip every GPU import.

### Interpreting results

`[series-start]` and `[series-end]` log accepted START/END boundaries at the shared
processing operator, with UTC timestamps and the CUDA device and series ID. START includes
the image shape, announced image count, and raster dimensions, with `scan_source`
`header_appendix` or `config`. END includes
received image, placed position, and batch counts. These timestamps describe arrival at
the processing stage, after transport and any decompression; END does not imply that GPU
work has completed. There is no per-image or per-batch logging.

At each completed series, `[series-result]` reports received/placed images, continuity gaps,
batch sizes, BF/ADF totals, and delivered throughput. Its UTC timestamp and explicit
`processed_images` count are emitted after the GPU totals copy completes; `images` retains
the same count for compatibility. Images outside the raster still count as processed;
`positions` counts the images placed in the BF/ADF maps. `delivered_fps` uses host timestamps at
BF/ADF batch submission, from the first image batch to the last, excluding the first batch
from its numerator. GPU work may still be queued at either timestamp, so this rate does not
measure completed GPU processing throughput. Completed-series totals are copied to the CPU
before reporting. Compare the rate with separately recorded sender image counts, source
acquisition duration, bstream drop deltas, and NIC
counters. NIC sampling measures transport traffic and includes protocol overhead; it is not
a substitute for delivered image counts.

For sample pixel validation, `python3 -m live4d.validate` accepts an NPZ containing
`ids` (unique zero-based sample IDs, shape `(N,)`) and CPU-reference `frames` (uint16,
shape `(N, 96, 96)` or `(N, 192, 192)`, matching the acquisition). `--expected-frames`
is the total acquisition count, which may be larger than the number of reference samples.
The validator is ZMQ-only and forces finite mode:

```bash
docker run --rm --gpus all --network host --ulimit memlock=-1:-1 \
  -v "$PWD/pipeline.yaml:/etc/dectris.yaml:ro" \
  -v "$PWD/reference.npz:/data/reference.npz:ro" \
  --entrypoint python3 dectris-holoscan:hsdk5 \
  -m live4d.validate --config /etc/dectris.yaml \
  --reference /data/reference.npz --expected-frames 1000000
```

Use a ZMQ configuration and replace the count with your acquisition's count. Start the
sender acquisition after `[holoscan] ready`. Exit status 0 requires the expected received
count, no continuity gaps, all supplied samples seen and pixel-exact, and completed
termination. It does not compare every pixel unless the reference includes every frame,
or independently validate BF/ADF sums. `--report` can save its JSON to a writable mount.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Docker cannot map `/dev/infiniband` | For ZMQ, remove the device argument or Compose device entry. For RDMA, verify host verbs devices first. |
| RDMA fails before readiness | Verify the sender handshake service, remote host/port, receiver RoCE address/device, locked-memory limit, and ring registration support. `host_mapped` is an alternative when GPU registration is unavailable. If imagemux closes the connection after the Hello, check that `rdma.protocol` matches the imagemux build. |
| Ready, but no series result | Confirm sender transport/compression and endpoint, trigger acquisition after readiness, and check that the sender emits END. `seconds: 0` allows an indefinite wait. |
| Unsupported image or START error | Both routes require 96×96 or 192×192 uint16 images matching START metadata; ZMQ additionally requires BSLZ4. START metadata must fit within 16 KiB. |
| Fewer placed images than received | Check the series' `scan_width * scan_height` against image IDs; out-of-scan IDs are intentionally excluded from maps. |
| Missing modules or old behavior after edits | Build the unified image again; source-only `PYTHONPATH` does not supply native operators or generated schemas. |

When recording a benchmark, retain the Git revision, image identity, GPU/NIC and topology,
sender version and settings, receiver YAML, run duration, count checks, and raw counters.
Keep historical measurements tied to their tested revision: the tensor-retirement change
in `4b4b12f` changed `delivered_fps` from per-batch GPU-completion timing to host-submission
timing. Older and newer rates need that distinction when compared. Transport-only RDMA
WRITE or NIC-loopback measurements do not establish Live4D processing capacity.

### Deferred known issue: intermittent RDMA tensor-pool exhaustion

Status: low priority; investigation is deferred. Revisit if it recurs in normal camera
acquisitions or when extending support for unusual acquisition sequences.

During some aggregate integration-test runs, the native RDMA source failed to borrow an
output tensor from its four-slot runtime pool with
`RDMA data tensor allocation failed: kResourceExhausted`. HSDK then stopped the graph.
The failing fixture is
`test_consecutive_empty_and_partial_series_keep_service_source_active` in
[test_rdma_batches.py](../holoscan_pipeline/tests/test_rdma_batches.py). It queues every
transport completion before graph startup, then feeds an empty 96×96 series, three
192×192 images with output batch size 2, and another empty 96×96 series. This is a
same-host verbs-loopback test using one GPU and a fixed connection; the failure has not
been established as a GPU-switching problem.

The root cause remains unknown, and no runtime workaround or fixture change was retained.
The fixture passed unchanged in the final suite, and cross-host two-GPU RDMA switching
checks passed, including two acquisitions totaling 7,200,000 checked frames. Those
successful runs do not establish that the intermittent failure is fixed.

Local diagnostic evidence is retained under the untracked
`benchmark_results/20261024-zmq-routing/` directory: `REPORT.md` summarizes the issue;
`evidence/tests-1792819468595634683/run.log` records a failure, and
`evidence/pytest-final-finite/run.log` records the final passing suite. Preserve these
artifacts when moving the investigation to another machine.
