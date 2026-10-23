# Agent instructions

## Project and references

This project develops streaming algorithms for DECTRIS data using Holoscan
SDK 5. ZMQ and imagemux RDMA inputs feed a shared processing pipeline.
Keep numerical algorithms separate from transport and scheduling.

Read [README.md](README.md) and [the pipeline guide](docs/holoscan_basic_ingest.md)
before changing the pipeline. Use this repository's documented build and test
commands.

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

After development changes, run focused correctness checks, then stress-test
the full configured pipeline at 120,000 fps with detector 2×2 binning for
30 seconds: 3,600,000 source frames, excluding warm-up and subsequent drain.
Stress testing checks sustained processing, correctness, loss, backlog,
resource growth, and clean completion.

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
