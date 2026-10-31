"""Send a .dstream capture over one ZMQ PUSH socket at a set frame rate, `repeats` times as
consecutive series, without parsing messages (the capture's first record is START, its last END).
PUSH blocks rather than drops when the receiver lags, so an achieved rate below the requested one
means the receiver could not keep up.

    python3 holoscan_pipeline/tests/benchmarks/paced_send.py <capture.dstream> \
        <endpoint to bind, e.g. tcp://127.0.0.1:5555> <rate Hz> <repeats>

It holds the capture in memory, so it reaches rates `live4d.parallax.tools.replay_to_zmq` cannot
(its paced mode parses each message). It needs live4d and pyzmq, as in the parallax image.
"""
import json, sys, time

import zmq

from live4d.parallax.ingest.dstream import iter_records

path, endpoint, rate, repeats = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
records = list(iter_records(path))
start, images, end = records[0], records[1:-1], records[-1]
sock = zmq.Context.instance().socket(zmq.PUSH)
sock.bind(endpoint)
time.sleep(1.0)                                   # let the receiver connect
results = []
for r in range(repeats):
    sock.send(start)
    t0 = time.perf_counter()
    for k in range(0, len(images), 64):
        for raw in images[k:k + 64]:
            sock.send(raw)
        ahead = (k + 64) / rate - (time.perf_counter() - t0)
        if ahead > 0:
            time.sleep(ahead)
    took = time.perf_counter() - t0
    sock.send(end)
    results.append(dict(series=r, frames=len(images), seconds=round(took, 3),
                        achieved_hz=round(len(images) / took, 1)))
    print(json.dumps(results[-1]), flush=True)
    time.sleep(1.0)
time.sleep(2.0)
sock.close(linger=2000)
