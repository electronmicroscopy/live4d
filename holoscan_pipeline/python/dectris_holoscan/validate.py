"""Validate a DAQ replay against CPU reference samples from the original HDF5 data."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cupy as cp
import numpy as np

from .app import BasicIngestApp, BasicIngestConfig, StatsSinkOp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True, help='NPZ with ids and frames arrays')
    parser.add_argument('--expected-frames', type=int, required=True)
    parser.add_argument('--endpoint', default='tcp://127.0.0.1:31001')
    parser.add_argument('--batch-size', type=int, default=1000)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--plan', type=Path)
    args = parser.parse_args()
    with np.load(args.reference) as reference:
        samples = dict(zip(reference['ids'].tolist(), reference['frames']))
    if args.expected_frames < 0 or any(not 0 <= i < args.expected_frames for i in samples):
        parser.error('reference IDs must lie within expected frame range')
    state = dict(frames=0, batches=0, first_id=None, last_id=-1, gap_events=0,
                 checked=set(), mismatched=[], started=None, finished=None, first_batch_size=0)

    class ValidationSink(StatsSinkOp):
        def process_batch(self, batch, ctx):
            ids = np.asarray(batch.image_ids, dtype=np.int64)
            if not len(ids):
                return
            now = time.perf_counter()
            if state['started'] is None:
                state.update(started=now, first_id=int(ids[0]), first_batch_size=len(ids))
            state['gap_events'] += int(ids[0] != state['last_id'] + 1)
            state['gap_events'] += int(np.count_nonzero(np.diff(ids) != 1))
            state['frames'] += len(ids)
            state['batches'] += 1
            state['last_id'] = int(ids[-1])
            if batch.frames.shape != (len(ids), 96, 96):
                raise AssertionError(f'Unexpected Tensor shape: {batch.frames.shape}')
            if batch.frames.__dlpack_device__() != (2, 0):
                raise AssertionError('Expected CUDA device 0 Tensor')
            matching = [(i, np.flatnonzero(ids == i)) for i in samples if ids[0] <= i <= ids[-1]]
            if matching:
                # Keep DLPack conversion on the SDK stream; only the correctness
                # sink copies device pixels to the host.
                with ctx.stream():
                    pixels = cp.asnumpy(cp.from_dlpack(batch.frames))
                if pixels.dtype != np.uint16:
                    raise AssertionError(f'Unexpected Tensor dtype: {pixels.dtype}')
                for image_id, positions in matching:
                    if len(positions):
                        state['checked'].add(image_id)
                        if not np.array_equal(pixels[int(positions[0])], samples[image_id]):
                            state['mismatched'].append(image_id)
            state['finished'] = time.perf_counter()
            if state['batches'] % 100 == 0:
                print(f"[validation] frames={state['frames']} gaps={state['gap_events']}", flush=True)

    app = BasicIngestApp(BasicIngestConfig(endpoint=args.endpoint, batch_size=args.batch_size),
                         sink_type=ValidationSink)
    if args.plan:
        args.plan.write_text(app.plan.json())
    termination = app.run()
    elapsed = (state['finished'] - state['started']) if state['started'] is not None else 0
    success = (state['frames'] == args.expected_frames and state['gap_events'] == 0
               and len(state['checked']) == len(samples) and not state['mismatched']
               and termination.name in ('REQUESTED_COMPLETED', 'COMPLETED'))
    report = dict(success=success, frames=state['frames'], batches=state['batches'],
                  first_id=state['first_id'], last_id=state['last_id'],
                  gap_events=state['gap_events'], cpu_reference_samples_checked=len(state['checked']),
                  mismatched_sample_ids=state['mismatched'], termination=termination.name,
                  elapsed_between_batches_seconds=elapsed,
                  fps=(state['frames'] - state['first_batch_size']) / elapsed if elapsed else None)
    text = json.dumps(report, indent=2)
    if args.report:
        args.report.write_text(text + '\n')
    print('[validation-result] ' + text, flush=True)
    raise SystemExit(0 if success else 1)


if __name__ == '__main__':
    main()
