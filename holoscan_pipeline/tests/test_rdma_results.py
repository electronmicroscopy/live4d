"""GPU result lifetime and completion without blocking another device's work."""
import time
import weakref

import cupy as cp
import numpy as np
import pytest

from live4d.rdma import ResultSlot


def test_result_slot_retains_storage_and_bounds_pending_work():
    slot = ResultSlot(0)
    stream = cp.cuda.Stream(non_blocking=True)
    finished = []
    with stream:
        totals = cp.asarray([123.0, 456.0], dtype=cp.float64)
        with pytest.raises(RuntimeError, match="reserved"):
            slot.submit(totals, finished.append)
        slot.available.clear()
        slot.submit(totals, finished.append)
        reference = weakref.ref(totals)
        with pytest.raises(RuntimeError):
            slot.submit(totals, finished.append)
    del totals
    assert reference() is not None
    deadline = time.monotonic() + 10
    while not slot.poll():
        assert time.monotonic() < deadline
    assert finished == [[123.0, 456.0]]
    assert slot.available.is_set()
    assert reference() is None
    assert not slot.poll()
    slot.close()


def test_result_on_second_device_completes_while_first_is_pending():
    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("requires two visible GPUs")
    slots = [ResultSlot(0), ResultSlot(1)]
    streams, owners, finished = [], [], []
    hold = cp.RawKernel(r'''
        extern "C" __global__ void hold(unsigned long long cycles) {
            unsigned long long start = clock64();
            while (clock64() - start < cycles) { }
        }
    ''', "hold")
    try:
        for device, slot in enumerate(slots):
            with cp.cuda.Device(device):
                stream = cp.cuda.Stream(non_blocking=True)
                streams.append(stream)
                with stream:
                    totals = cp.asarray([device + 1.0, 17.0])
                    owners.append(totals)
                    if device == 0:
                        # Compile before the measured pending interval.
                        hold((1,), (1,), (np.uint64(1),))
                        stream.synchronize()
                        rate = cp.cuda.runtime.getDeviceProperties(device)["clockRate"]
                        hold((1,), (1,), (np.uint64(rate * 1000),))
                    slot.available.clear()
                    slot.submit(totals, lambda value, d=device: finished.append((d, value)))
        deadline = time.monotonic() + 10
        while not slots[1].poll():
            assert time.monotonic() < deadline
        assert not slots[0].poll()
        assert finished == [(1, [2.0, 17.0])]
    finally:
        for slot in slots:
            slot.close()
