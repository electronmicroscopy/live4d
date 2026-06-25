# Real-Time STEM Virtual Bright-Field

Live virtual bright-field (VBF) from 4D-STEM frames streamed off a DECTRIS detector. The detector
sends compressed frames over ZMQ; the pipeline decompresses them on the GPU (nvCOMP + a custom CUDA
kernel) and accumulates the VBF image on the GPU as frames arrive. Current rate ~22 kHz end-to-end at
192² unbinned (targets: 30 kHz unbinned, 120 kHz binned). Next step: migrate to Holoscan / C++ to
reach full detector rate.

VBF is a proof-of-concept workload to exercise the streaming + GPU-decode path end-to-end, not the
final method. The pipeline is built to carry heavier algorithms (parallax / tcBF) next.

## Approach

- **GPU decode.** bslz4 (bitshuffle + LZ4) is decompressed on the GPU: batched nvCOMP LZ4 for all
  blocks at once, then a custom CUDA bit-unshuffle kernel. The CPU only parses the small CBOR
  envelope and ships the compressed bytes over.
- **Threaded chunked pipeline.** Frames move in chunks through producer → decoder pool → consumer,
  keeping per-frame queue/GIL overhead off the hot path.
- **GPU-side compute.** Integer frames upload as-is; the int→float cast and the mask·sum reduction
  run on the GPU (the host-side float cast had been a major cost).
- **Two decode backends**, chosen at runtime: `cpu` (threaded host bslz4) or `gpu`.

## Layout

```
pipeline/                  streaming pipeline
  ingest/                  data in: transport, CBOR decode, GPU decompression
  virtual_detectors/       VBF engine, config, main.py
  visualize/               image display
  docs/stream_format.md    DECTRIS transport + bslz4 reference
simple_listener/           ~100-line reference: connect, decode CBOR, count frames
```

## Data path

```
DCU ─ZMQ PULL─▶ CBOR msg ─tag 56500─▶ bslz4 (bitshuffle+LZ4) ─decompress─▶ frame ─mask·sum (GPU)─▶ VBF
```

CBOR over a ZMQ PULL socket; image payload under tag 56500, compressed with bslz4 (Masui bitshuffle,
HDF5 filter 32008, per-block LZ4). VBF: `S = Σ frame·W`, where `W` = bright-field disk × dead-pixel
mask × flatfield. Frames carry an image id giving the scan position. Format details in
`pipeline/docs/stream_format.md`.

## From the simple listener

The listener's CBOR decode (`tag_hook` + `tag_decoders`) is reused almost verbatim in
`ingest/dectris_decode.py`. Added around it: the threaded chunked pipeline and the VBF compute.
Key change: the listener decompresses on the CPU inside `cbor2.loads`; the GPU path
parses only the envelope on the CPU (`decode_envelope`) and decompresses on the GPU (`gpu_decode.py`).
That split takes end-to-end from ~5 to ~22 kHz.

## Components (`pipeline/`)

| file | role |
|---|---|
| `ingest/dectris_decode.py` | CBOR decode: `decode_message` (parse + decompress, CPU); `decode_envelope` (envelope only, GPU path). |
| `ingest/bslz4.py` | bslz4/lz4 container framing parser (header + per-block offsets). |
| `ingest/gpu_decode.py` | GPU decoder: batched nvCOMP LZ4 + custom CUDA unshuffle kernel. Self-test: `python -m ingest.gpu_decode`. |
| `ingest/pipeline.py` | threaded chunked pipeline (producer → decoder pool → consumer); cpu/gpu backends; stats. |
| `virtual_detectors/vbf.py` | VBF engine; int→float cast + reduce on the GPU. |
| `virtual_detectors/config.py` | detector/scan geometry, BF disk, pipeline knobs. |
| `virtual_detectors/main.py` | entry point (live). |
| `visualize/` | image display. |

## Performance

NVIDIA IGX Orin (aarch64) + RTX 6000 Ada over PCIe, on beam-on ARINA data:

| stage / mode | rate |
|---|---|
| GPU decode, isolated | ~80–96 kHz |
| CPU CBOR envelope parse | ~50 kHz |
| end-to-end, `--decode cpu` (192²) | ~5 kHz |
| end-to-end, `--decode gpu` (192² / 96² binned) | ~22 kHz |

## Running

```
cd pipeline
python -m ingest.gpu_decode                                   # validate GPU decoder (vs CPU reference)
python virtual_detectors/main.py --headless --decode gpu      # headless run + per-stage timing
python virtual_detectors/main.py                              # live view (DCU from config.py)
```

`main.py`: `--decode cpu|gpu`, `--binned`, `--chunk-size`, `--decode-threads`, `--headless`. The
GPU path needs system CUDA (`nvcc`) + `ninja` for the runtime kernel build
(`CUDA_HOME=/usr/local/cuda`, `TORCH_CUDA_ARCH_LIST=8.9` for Ada). Dependencies in
`pipeline/requirements.txt`.
