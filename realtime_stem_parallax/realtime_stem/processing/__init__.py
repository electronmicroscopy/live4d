"""Processing stage: turn batches of decoded frames into a reconstruction.

Every algorithm implements two small contracts (`processing.base`):
  * `Accumulator` — the hot path: `process(frames, ids)` folds a batch into the
    reconstruction-so-far; `result()` returns it. The ingest sink talks only to this.
  * `Algorithm` — the wiring: CLI flags, building the accumulator, saving results.

`processing.registry` maps a `--algorithm` name to its `Algorithm`. Parallax / tcBF is the
worked example (`processing.parallax`); adding a method means implementing these two contracts.
"""
