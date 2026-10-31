"""Transport only: count a DCU's (or daqsim's) Stream2 messages on one PULL connection, doing
nothing else, until the stream has been idle for 5 s. Several running at once count over several
connections, since the stream hands its messages out in turn.

    python3 holoscan_pipeline/tests/benchmarks/zmq_probe.py tcp://127.0.0.1:31001 out.json
"""
import json, sys, time
import zmq

ctx = zmq.Context()
sock = ctx.socket(zmq.PULL)
sock.connect(sys.argv[1])
count = nbytes = 0
first = last = None
small = []
while True:
    if not sock.poll(5000 if count else 120000):
        break
    msg = sock.recv(copy=False)
    now = time.perf_counter()
    first = first if first is not None else now
    last = now
    count += 1
    nbytes += len(msg)
    if len(msg) < 1024:
        small.append(count)
result = dict(messages=count, bytes=nbytes, seconds=None if first is None else last - first,
              rate=None if not count or last == first else (count - 1) / (last - first),
              small_messages=len(small))
print(json.dumps(result), flush=True)
open(sys.argv[2], "w").write(json.dumps(result))
