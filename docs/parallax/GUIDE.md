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
`live_every` or `live_every_rows` throttle them. Live tuning requests enter a bounded queue and apply in the next decoded callback
on its runtime CUDA stream; HTTP threads never retain borrowed stream handles. The right panel's throughput and host batch timing refer to the
**parallax branch**. Host submission timings include existing waits; they measure neither
GPU kernel duration nor whole-graph completed throughput. CPU/GPU/network resources
are process/host observations. BF/ADF images are outside this viewer's scope.

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
