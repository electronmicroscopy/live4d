# Parallax in Live4D

`python -m live4d --config pipeline.yaml` runs one HSDK5 application. BF/ADF always
processes the decoded batches. Optional parallax, HDF5, and ANS consumers share those
same read-only tensors, bounded by the slowest consumer. Each branch has its own
partition, and finite acquisition completes only after all enabled branches finish END.

Build either target from the repository root (set CUDA architecture for your GPU):

```bash
docker build -f docker/Dockerfile.holoscan --target pipeline --build-arg CUDA_ARCHITECTURES=86 -t live4d:pipeline .
docker build -f docker/Dockerfile.holoscan --target parallax --build-arg CUDA_ARCHITECTURES=86 -t live4d:parallax .
```

Both targets launch `python3 -m live4d`. The parallax target adds CUDA PyTorch and
offline decoding dependencies. YAML selects the branches. ANS additionally needs
`quantem.gpu` (see `docker/Dockerfile.ans`; it can extend `live4d:parallax`). The source
wheel supports offline utilities; native bindings and schemas for live use are built
by CMake and included in the Docker targets. Viewer assets are included in both.

Parallax defaults to `enabled: false`, and does not import PyTorch when disabled.
When enabled, these acquisition values are required under `parallax`:

| Key | Unit |
| --- | --- |
| `energy` | eV |
| `semiangle` | mrad |
| `angular_sampling` | mrad per input detector pixel, for the actual binning |
| `scan_sampling` | Å per scan pixel |
| `defocus` | Å, negative for underfocus in the established convention |
| `rotation` | degrees |

There are no production defaults for these values. `scan.width/height`, `device`,
`batch_size`, transport, and queues come from shared settings. START metadata selects
96×96 or 192×192 uint16 geometry and may override the scan dimensions. A new scan-mode
reconstruction is built per START. Persistent modes reject changes to scan or detector
geometry; their acquisition calibration must remain appropriate for the detector mode.

Other parallax settings include `center: [row, col]` (default: frame center), `c12`
(Å), `phi12` (degrees), `upsampling`, `flip_cols`, `detector_layout`,
`subtract_frame_mean` (false), `phase_flip`, `dc_correct`, `normalize`, `coverage`,
`lowpass`, `highpass`, `butterworth_order`, `suppress_nyquist: auto|on|off`, and
`band_limit`. Filters are readout operations. `accumulate` accepts `scan`, `latest`,
`all`, `reset`, or `swap`; `latest` requires batch size no larger than scan positions.
`swap` supports `pass_average` in (0,1]. `save_passes` and `save_passes_dtype` save
completed passes; writes apply backpressure so pending output memory stays bounded.

`out_dir` receives the existing raw, corrected, phase-flipped, and optional coverage
arrays, shift arrays, and a numbered copy for each acquisition. END reports completeness.
Interruption saves partial results and reports incompleteness without fabricating END.
Service mode keeps one connection and graph alive across scans. `scan` starts fresh;
the other accumulation modes persist, including the existing pass numbering behavior.

Set `viewer_port` (for example 8050) and optionally `viewer_host` to serve the browser
viewer with live tuning. `live_endpoint` optionally publishes snapshots over ZMQ;
`live_every` or `live_every_rows` throttle them. Live tuning requests enter a bounded queue and apply in the next parallax callback
on its runtime CUDA stream: a decoded batch, or in service mode the branch's tick between batches. HTTP threads never retain
borrowed stream handles. The live benchmarks panel on the right is for debugging and opens while
the viewer's "live benchmarks" checkbox is ticked, or with `?benchmarks=1` in the address. Its
throughput and host batch timing refer to the **parallax branch**. Host submission timings include existing waits; they measure neither
GPU kernel duration nor whole-graph completed throughput. CPU/GPU/network resources
are process/host observations. BF/ADF images are outside this viewer's scope.

## Optimum bright field (OBF)

With `accumulate: latest`, OBF direct ptychography of the frames the store holds is a live channel
next to parallax, on the parallax output grid. The viewer's channel list holds parallax,
phase-flipped parallax (the default) and OBF. OBF is reconstructed only while it is shown, so
choosing it starts the reconstruction and choosing another channel stops it. These settings sit
under `parallax`:

| Key | Default | Effect |
| --- | --- | --- |
| `compute_obf` | `false` | Reconstruct OBF from the start, as choosing it in the viewer does. It adds the channel `obf` and saves `parallax_obf.npy`. |
| `precompute_obf_weights` | `false` | Build OBF's weights (its Gram) from the start and again after every retune, so that choosing OBF needs only a sweep of the frames held. The viewer's "OBF weights" checkbox does the same. |
| `obf_share` | `0.4` | The share of the parallax branch's time OBF may take, on its stream. |

Both OBF switches require `accumulate: latest`, since the other modes keep no frames to read.

OBF costs much more per frame than parallax, so it never runs inside the parallax accumulation.
After each batch the sink sets aside `obf_share` of the interval between batches (within 0.5 to
50 ms) for OBF, so OBF's part of the branch follows the frame rate. It saves these shares up into
steps of 8 ms, so that a step's own cost of gathering positions and launching the deposit stays a
small part of it. In the live graph the interval is the one between the batches' publish times,
which the source sets. A batch that waited in the queue longer than that interval means the branch
is behind, and it adds nothing to OBF's share, nor does the batch that publishes a snapshot. OBF
also yields when the branch is busy for any other reason, such as the viewer's server threads. The
sink compares the branch's time per batch with the interval over half a second, halves OBF's share
when it exceeds 80%, down to none, and lets it grow back below 60%. A stalled branch fills the
receiver's publication window, and then the source's buffers, within tens of milliseconds, so the
branch has to keep up batch by batch and not only on average. The offline replay has no publish
times and uses the time since the previous batch, so there OBF slows the replay by about its share.
The step runs inside the parallax operator's callback, on its runtime stream, and the tracker
(`live4d.parallax.obf`) keeps no thread and no stream of its own. Each step's device work is
bracketed by CUDA events. The next step does nothing until they have completed, and sizes its work
from the longer of its device time and its host time, so at most one step is queued behind the
parallax work. Its weights take seconds per geometry and are built in the same steps, in pieces of
about 10 ms of host or device work (`ObfSetup.build_steps`). The first build in a run also creates
cuFFT plans, which stalls the branch once for up to about 0.8 s. A deposit pads its batches to
powers of two, so that its FFTs see a few sizes, since a cuFFT plan for a new size takes tens of
milliseconds to make. The last three setups are cached, so a retune back to an earlier geometry
finds its weights again. OBF keeps up with the frames when its deposit rate times its share exceeds
the frame rate, and a larger share leaves the parallax branch less headroom.

In service mode the parallax branch also runs on a 20 Hz clock (`ParallaxTickOp`) between
batches. Each tick applies queued tuning, and once no batch has come for two ticks it gives OBF
`obf_share` of the period, so C10, C12 and the rotation can be tuned between scans and after the
last one, and OBF follows the retune. A finite
run finishes OBF at END, before it saves, so `parallax_obf.npy` is then the image of every frame
held with exact bright-field means. In service mode the next START may follow at once, so END
saves OBF's latest image as it is, and the run's end saves it again if the ticks refined it.

Choosing OBF shows an image at once and lets OBF wipe in, as a scan would. The channel starts as the
phase-flipped parallax image of the same frames, scaled to OBF's contrast, and OBF replaces it row
by row from the scan line outwards. New frames are deposited as they arrive, and the frames held
are swept back from the scan line, newest first, with the budget left. A covered position the scan
writes again is updated by its new values minus those deposited before, so OBF keeps its own copy
of the store, as much device memory again as the store. When OBF is slower than the frames it lags
and deposits each position once, at its newest frame. A dot next to the checkbox shows the
weights' state and a dot next to the channel list shows OBF's, with the details on hover. The
snapshot metadata carries `obf_state`, `obf_weights_state`, `obf_coverage`, `obf_age_s`,
`obf_rate_khz`, `obf_sweep_s` and, while OBF yields to a busy branch, `obf_share_scale`.

The method is the correlation route of quantem's `DirectPtychographyMontage`, ported to torch in
`live4d/algorithms/obf/core.py`. The probe is built on the parallax geometry (the rotation, the
column flip, the disk centre, C10 and C12), so OBF and parallax agree in orientation by
construction. Each bright-field pixel is normalized as quantem does, V = I/m − 1, with m its mean
over the frames held when the wipe starts. The patches hold only the band the output grid can
show, which on the binned Gd₂O₃ recording at ×2 makes a frame 3.8 times cheaper and changes the
image by 0.3%. The port matches quantem's correlation route to 2e-6 relative on identical inputs,
and quantem's own OBF of the 2026-10-20 Gd₂O₃ recordings to a correlation of 0.999 away from the
scan edges (EM_Realtime_Processing, `sessions/2026-10-20_Gd2O3/analysis/obf_vs_quantem.py`).

Per frame, the correlation route correlates the frame with the probe for every sub-lattice of the
patch grid and keeps only the lags the patch holds. The FFT route that quantem and `obf-live` use
forms each sub-lattice's spectrum over the padded detector grid, transforms it back in full and
discards most of it, and it computes a second detector FFT that is the first one's conjugate
reflected (the values are real). Here a frame takes one detector FFT and two matrix products
(`_PatchProducts`). The first multiplies the transform's real and imaginary parts by weights that
combine the partial inverse transform with the probe operands, and the second completes the
partial inverse transform along the other axis. The result equals the FFT route to 2e-14 in
float64 and 1e-6 in float32. The lattice placement, used for X-ray data, keeps the FFT route.

`python -m live4d.parallax.scripts.bench_obf` times the setup, the deposit rate, where a batch's
time goes and one readout. On an A40, for a 256² scan, the weights build in a single call and
the deposits run back to back:

| Detector | Upsampling | Weights | Deposit | Sweep of the scan | Readout | `obf-live` deposit |
| --- | --- | --- | --- | --- | --- | --- |
| binned 96² | ×2 | 0.8 s | 415 kHz (2.4 µs/frame) | 0.2 s | 1.4 ms | 131 kHz |
| unbinned 192² | ×2 | 1.6 s | 85 kHz (11.8 µs/frame) | 0.8 s | 1.5 ms | 32 kHz |
| binned 96² | ×4 | 1.5 s | 149 kHz (6.7 µs/frame) | 0.4 s | 5.2 ms | 68 kHz |
| unbinned 192² | ×4 | 4.0 s | 32 kHz (31 µs/frame) | 2.0 s | 5.0 ms | 20 kHz |

With the default share of 0.4, OBF at ×2 therefore keeps up with the detector's 120 kHz binned
and its 20 to 24 kHz unbinned. At ×4 it falls behind during a scan and covers the frames held
after it. While frames stream, its weights take their build time divided by the share, about 2 s
binned at ×2 (four passes of a 256² scan at 120 kHz), before OBF covers any frame. A finite run's
finish covers the frames held, and `precompute_obf_weights` builds the weights before OBF is
chosen, for example while focusing with parallax.

Live on two A40 hosts, daqsim streamed the binned Gd₂O₃ recording over ZMQ with the viewer open,
its weights switched on and OBF chosen during the scan. At 50 kHz, at ×2 and at ×4, no frame was
dropped, the parallax branch kept the source's rate, and OBF covered the frames held within about
8 s of being chosen. Above 50 kHz one daqsim stream connection is the limit, at about 57 thousand
of these frames a second. A sender of our own on one socket, without daqsim, ran the recording at
100 kHz at ×2 and ×4 and at 120 kHz at ×4, with OBF on and the viewer's server publishing. Every
frame arrived at the sender's rate and OBF was live from the third pass, at 120 kHz with its share
scaled to 0.625.

OBF looks grainier and flatter than phase-flipped parallax, though it shows the lattice at least
as well. It whitens the noise up to the band and transfers little at low frequencies, and a
low-pass of about 1.5 Å in the viewer suits it. Within a probe's width of the scan edges OBF
differs from parallax, since its canvas does not wrap. It uses parallax's bright-field pixels, so
the soft edge of the aperture beyond the semiangle is left out. Like parallax it models C10 and
C12 only, with an analytic probe and no partial coherence.

## At the microscope

On the computer the ARINA's DCU streams to, build the `parallax` target for its GPU, for example
`CUDA_ARCHITECTURES=90` for an H100 or H200, 89 for Ada, 86 for an A40 and 121 for a DGX Spark:

```bash
git clone https://github.com/electronmicroscopy/live4d && cd live4d
docker build -f docker/Dockerfile.holoscan --target parallax \
  --build-arg CUDA_ARCHITECTURES=90 -t live4d:parallax .
```

The DCU's stream must be enabled, with format `cbor` and compression `bslz4`. Write
`pipeline.yaml` with the DCU's Stream2 endpoint, the scan and the acquisition's own values. The
values below are the 2026-10-20 Gd₂O₃ session's calibration, to show the keys, not defaults.

```yaml
transport: zmq
batch_size: 512
execute:
  mode: service            # one graph across scans; Ctrl-C saves and stops
scan:
  width: 256
  height: 256
zmq:
  endpoint: tcp://DCU_ADDRESS:31001
parallax:
  enabled: true
  energy: 200000           # eV
  semiangle: 30            # mrad
  angular_sampling: 1.066  # mrad per pixel, binned 2x2
  scan_sampling: 1.9766    # Å
  defocus: -281            # Å
  rotation: 4.78           # degrees
  center: [47.86, 48.41]
  flip_cols: true
  c12: 17
  phi12: -7.1
  upsampling: 2
  accumulate: latest       # needed for OBF
  precompute_obf_weights: true   # OBF's weights ready before it is chosen
  compute_obf: false       # true: reconstruct OBF from the start, without the viewer
  viewer_port: 8050
  out_dir: /data
```

```bash
docker run --rm --gpus all --network host --ulimit memlock=-1:-1 --ulimit stack=33554432 \
  -v "$PWD/pipeline.yaml:/etc/live4d.yaml:ro" -v "$PWD/out:/data" \
  live4d:parallax --config /etc/live4d.yaml
```

Arm the detector after `[holoscan] ready`. The viewer is at `http://127.0.0.1:8050` on that
computer, or on another through `ssh -L 8050:127.0.0.1:8050 that-computer`. One ZMQ connection
carried about 55 thousand real binned frames a second from daqsim (see
[the validation records](VALIDATION.md)), and the DCU also paces each connection, so faster scans
lose frames at the source until the receiver takes several connections. RDMA
(`holoscan_pipeline/config/rdma.yaml`, with `--device /dev/infiniband` added to `docker run`)
avoids that limit.

## Migrate an existing acquisition

Keep the original `site.flags` and recordings. Convert explicitly into a new YAML file:

```bash
python -m live4d.parallax.tools.migrate_flags --flags site.flags --output pipeline.yaml
```

This accepts the existing physics, readout, orientation and accumulation flags, maps
`--scan NY NX` to shared scan height/width, and preserves explicit angular sampling.
It does not guess missing acquisition values or overwrite existing YAML. Edit the new
file to add the transport endpoint/device and desired viewer settings. `--binned` is
replaced by START geometry; its explicitly supplied angular sampling is preserved.
The old multi-connection and scheduler flags have been removed.

Replace the previous Docker command that invoked the standalone runner with:

```bash
docker run --rm --gpus all --network host \
  -v "$PWD/pipeline.yaml:/etc/live4d.yaml:ro" \
  -v "$PWD/out:/data" live4d:parallax --config /etc/live4d.yaml
```

Set `parallax.out_dir: /data` to retain results. RDMA additionally requires the verbs
device, memlock limit, and shared RDMA YAML settings described in the
[pipeline guide](../holoscan_basic_ingest.md). No detector settings are changed by Live4D.
Single-connection performance must be measured on the target machine.

## Offline replay

```bash
pip install -e './holoscan_pipeline[parallax,dev]'
python -m live4d.parallax.offline --config pipeline.yaml --source capture.dstream --device cpu
python -m live4d.parallax.offline --config pipeline.yaml --source frames.npy --device mps
```

Offline replay imports no Holoscan and uses the same YAML, numerical implementation,
sink, tuning and viewer. Offline `.dstream` replay accepts the uint16/uint32 dtype announced by START;
live HSDK5 ingest remains uint16. `.dstream` files preserve consecutive acquisitions; `.npy`
arrays provide one series in raster order. `--source-rate` optionally paces replay.
When enabled, the viewer stays available after replay until Ctrl-C. Tools, examples,
and benchmarks are modules under `live4d.parallax.tools`, `.examples`, and `.scripts`;
see the generated [CLI reference](CLI.md).

Run correctness tests with `python -m pytest holoscan_pipeline/tests/parallax` and
shared live tests with `python -m pytest holoscan_pipeline/tests dectris/tests` inside
the appropriate image. Live tests require CUDA and native bindings. A stress result
must record completed processing, loss/completeness, backlog and memory, detector
binning/geometry, actual sender rate, and drain; a configured 120,000 fps alone is
insufficient evidence. See [validation results](VALIDATION.md).
