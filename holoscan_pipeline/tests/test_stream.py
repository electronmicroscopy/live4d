"""GPU integration regressions using real SmB6 BSLZ4 chunks on a private ZMQ port.

Run on the Docker/GPU host after building :hsdk5. The local DAQ simulator is
not involved, so these checks also cover empty and truncated final batches.
"""
import json
import os
from pathlib import Path
import subprocess
import uuid

import cbor2
import h5py
import numpy as np
import pytest
import zmq

ROOT = Path(__file__).resolve().parents[2]
IMAGE = os.environ.get('HOLOSCAN_TEST_IMAGE', 'dectris-holoscan-basic:hsdk5')
DATA = Path(os.environ.get('SMB6_TEST_MASTER', ROOT / 'daqsim/input/SmB6_preview_master.h5'))
REFERENCE = Path(os.environ.get('SMB6_TEST_REFERENCE', ROOT / 'daqsim/holoscan-reference.npz'))
pytestmark = pytest.mark.skipif(not DATA.exists() or not REFERENCE.exists(),
                                reason='local SmB6 data/reference not available')


def start_container(tmp_path, endpoint, *args):
    name = 'stanford-hs5-test-' + uuid.uuid4().hex[:10]
    process = subprocess.Popen([
        'docker', 'run', '--rm', '--name', name, '--gpus', 'all', '--network', 'host',
        '-e', 'HOLOSCAN_LOG_LEVEL=WARN', '-v', f'{tmp_path}:/validation',
        '--entrypoint', 'python3', IMAGE, *args,
        '--endpoint', endpoint], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return name, process


def finish(name, process):
    subprocess.run(['docker', 'rm', '-f', name], capture_output=True)
    if process.poll() is None:
        process.communicate(timeout=10)


@pytest.mark.parametrize('count', [0, 1, 128, 257])
def test_order_pixels_and_end_of_stream(tmp_path, count):
    with h5py.File(DATA) as source:
        mask, chunk = source['entry/data/data_000001'].id.read_direct_chunk((0, 0, 0))
        assert mask == 0
    with np.load(REFERENCE) as reference:
        frame = reference['frames'][list(reference['ids']).index(0)]
    ids = sorted({0, count // 2, count - 1}) if count else []
    np.savez(tmp_path / 'reference.npz', ids=np.array(ids, dtype=np.int64),
             frames=np.stack([frame] * len(ids)) if ids else np.empty((0, 96, 96), dtype=np.uint16))
    with zmq.Context() as context:
        sender = context.socket(zmq.PUSH)
        sender.setsockopt(zmq.SNDTIMEO, 15000)
        sender.setsockopt(zmq.LINGER, 0)
        port = sender.bind_to_random_port('tcp://127.0.0.1')
        name, process = start_container(
            tmp_path, f'tcp://127.0.0.1:{port}', '-m', 'dectris_holoscan.validate',
            '--reference', '/validation/reference.npz', '--expected-frames', str(count),
            '--batch-size', '128', '--report', '/validation/report.json')
        try:
            sender.send(cbor2.dumps({'type': 'start'}))
            payload = cbor2.CBORTag(40, [[96, 96], cbor2.CBORTag(69,
                       cbor2.CBORTag(56500, ['bslz4', 2, chunk]))])
            for image_id in range(count):
                sender.send(cbor2.dumps({'type': 'image', 'image_id': image_id,
                                        'data': {'threshold_1': payload}}))
            sender.send(cbor2.dumps({'type': 'end'}))
            output, _ = process.communicate(timeout=45)
            assert process.returncode == 0, output
            report = json.loads((tmp_path / 'report.json').read_text())
            assert report['success'], report
            assert report['batches'] == (count + 127) // 128
        finally:
            sender.close()
            finish(name, process)


def test_idle_sigterm(tmp_path):
    name, process = start_container(tmp_path, 'tcp://127.0.0.1:1', '-m', 'dectris_holoscan')
    try:
        # Readiness comes after the Python signal handlers and session are installed.
        for line in process.stdout:
            if '[holoscan] ready' in line:
                break
        else:
            pytest.fail('pipeline exited before readiness')
        subprocess.run(['docker', 'kill', '--signal', 'TERM', name], check=True, capture_output=True)
        output, _ = process.communicate(timeout=10)
        assert process.returncode == 0, output
    finally:
        finish(name, process)
