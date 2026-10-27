"""Validate decoded pixels and the shared processing output against CPU reference images."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import cupy as cp
import numpy as np

from .app import PipelineApp
from .config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--reference", type=Path, required=True, help="NPZ with ids and frames arrays")
    parser.add_argument("--expected-frames", type=int, required=True)
    parser.add_argument("--endpoint")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--plan", type=Path)
    args = parser.parse_args()
    with np.load(args.reference) as reference:
        samples = dict(zip(reference["ids"].tolist(), reference["frames"]))
    if args.expected_frames < 0 or any(not 0 <= i < args.expected_frames for i in samples):
        parser.error("reference IDs must lie within expected frame range")
    checked, mismatched, results = set(), [], []

    def observer(batch, positioned):
        ids = np.asarray(batch.image_ids, dtype=np.uint64)
        matching = [(i, np.flatnonzero(ids == i)) for i in samples]
        if any(len(positions) for _, positions in matching):
            pixels = cp.asnumpy(positioned.images)
            for image_id, positions in matching:
                if len(positions):
                    checked.add(image_id)
                    if not np.array_equal(pixels[int(positions[0])], samples[image_id]):
                        mismatched.append(image_id)
        if batch.end_of_series:
            results.append(positioned.result)

    config = load_config(args.config)
    overrides = dict(execute=config.execute | dict(mode="finite"))
    if args.endpoint:
        overrides["zmq"] = config.zmq | dict(endpoint=args.endpoint)
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    config = replace(config, **overrides)
    if config.transport != "zmq":
        parser.error("the sample validator uses ZMQ; use the common application for RDMA acceptance")
    app = PipelineApp(config, observer=observer)
    if args.plan:
        args.plan.write_text(app.plan.json(), encoding="utf-8")
    termination = app.run()
    result = results[0] if len(results) == 1 else {}
    success = (result.get("images") == args.expected_frames and result.get("gap_events") == 0
               and len(checked) == len(samples) and not mismatched
               and (not app.lane_mode or app.receiver.completed == 1)
               and termination.name in ("REQUESTED_COMPLETED", "COMPLETED")
               + (("STOPPED",) if app.lane_mode else ()))
    report = dict(success=success, **result, cpu_reference_samples_checked=len(checked),
                  mismatched_sample_ids=mismatched, termination=termination.name)
    output = json.dumps(report, indent=2)
    if args.report:
        args.report.write_text(output + "\n", encoding="utf-8")
    print("[validation-result] " + output, flush=True)
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
