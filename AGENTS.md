# Agent instructions

## Project and references

This project develops streaming algorithms for DECTRIS data using Holoscan
SDK 5. ZMQ and imagemux RDMA inputs feed a shared processing pipeline.
Keep numerical algorithms separate from transport and scheduling.

Both transports support ARINA uint16 images at 192×192 with 1×1 binning
(unbinned) and 96×96 with 2×2 binning. Each series' START metadata selects
the dimensions; consecutive series may change size. Runtime tensor shapes
follow the series, while pool capacity covers the larger images. BF/ADF
radii remain configured in input-image pixels; they do not scale automatically.

Read [README.md](README.md) and [the pipeline guide](docs/holoscan_basic_ingest.md)
before changing the pipeline. Parallax numerical code lives in
`holoscan_pipeline/python/live4d/algorithms/parallax/`; lifecycle, tuning, viewer,
telemetry, recording tools, and offline replay live in `live4d/parallax/`.
There is one live application (`python -m live4d`) and HSDK5 graph. Parallax is
optional and disabled by default; never supply guessed acquisition physics.
Offline replay (`python -m live4d.parallax.offline`) shares the YAML and numerical code.
Run migrated tests in `holoscan_pipeline/tests/parallax/`; regenerate CLI docs with
`python -m live4d.parallax.scripts.gen_cli_reference` after CLI changes.
Keep decoded tensors read-only; each enabled branch releases its own publication
consumer and joins the finite-mode completion latch. Keep batch PyTorch work and live tuning on the runtime stream inside active callbacks.
Queue HTTP tuning requests; never retain a runtime stream for another thread or callback.
After interrupted shutdown, synchronize before saving application-owned partial results.


Use the [Holoscan SDK 5 documentation](https://docs.nvidia.com/holoscan/sdk-user-guide/)
as the reference for Holoscan development. Select the documentation version
matching the SDK used by this project. Read the conceptual pages, API references,
and examples relevant to the task; verify unfamiliar signatures and ownership
contracts before using them.

Do not load the full documentation for every task or every agent. Give each
delegated agent the relevant references and constraints for its assignment.
Expand reading when a dependency, diagnostic, or uncertainty requires it.
Do not rely on remembered 4.x patterns when developing against HSDK5.

See also the SDK's [AI coding assistants guide](https://docs.nvidia.com/holoscan/sdk-user-guide/resources/ai-coding-agents)
and [tensor ownership and stream contract](https://docs.nvidia.com/holoscan/sdk-user-guide/data-and-payloads/tensors-in-depth).

## Development

Keep transport handling in `dectris/` and the native ingest operators.
Keep numerical algorithms under `live4d.algorithms` without Holoscan imports.
Processing operators own scheduling, scan placement, and GPU buffer lifetime.

Preserve typed messages, acquisition boundaries, valid batch lengths,
bounded buffering, tensor ownership, CUDA readiness, and RDMA credit ordering.
Preserve support for both image sizes and reject images that disagree with START.
Avoid hidden copies, allocations, and unnecessary synchronization in hot paths.

Use the compose → compile → run workflow. Inspect graph diagnostics before
runtime testing. A compiled graph does not prove runtime correctness or speed.

Keep changes focused. Extend existing tests that own the changed behavior.
Do not commit, push, publish, or deploy without explicit authorization.

## Testing

Follow [the pipeline guide](docs/holoscan_basic_ingest.md#validation-and-measurements)
for setup, validation commands, ownership contracts, metric interpretation, and
benchmark evidence.

Before each test group, explain its purpose, whether it is correctness,
smoke, or stress testing, and its pass criteria and limitations.

For image-geometry changes, focused correctness checks must cover both supported
sizes, size changes between series, empty and partial batches, and geometry
mismatches. Keep numerical reference checks independent of the implementation.

After development changes, run focused correctness checks, then stress-test
the full configured pipeline at 120,000 fps with detector 2×2 binning for
30 seconds: 3,600,000 source frames, excluding warm-up and subsequent drain.
Stress testing checks sustained processing, correctness, loss, backlog,
resource growth, and clean completion.

The 120,000 fps target applies to 96×96 images with 2×2 binning. Record binning
and image dimensions with each result; report unbinned throughput separately.

Prefer RDMA when available; otherwise use ZMQ. For shared pipeline changes,
test both when available. Transport-specific changes require the affected
path. Report unavailable paths as unverified and identify whether RDMA
testing was same-host or cross-host.

For same-host simulated testing, use the git-ignored `daqsim/` directory.
Verify its presence, version, supported transport, and acquisition settings
before running; record its command and configuration.

Use the guide's measurement and reporting requirements. A configured sender
rate, submission rate, or transport-only result does not prove completed
pipeline throughput. Report failures and missing evidence explicitly.
