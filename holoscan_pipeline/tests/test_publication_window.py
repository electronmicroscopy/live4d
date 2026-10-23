"""Exact batch-completion checks for the native publication window wrapper."""

import gc
import os
import weakref

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("RDMA_TEST_IN_IMAGE") != "1",
    reason="requires the HSDK 5 build image",
)


@pytest.fixture
def publication_window_type():
    from live4d.operators import _PublicationWindow

    return _PublicationWindow


def test_out_of_order_acknowledgement_does_not_advance_past_gap(publication_window_type):
    window = publication_window_type(2)
    try:
        acknowledge = window.consumer("default")
        acknowledge(2)
        with pytest.raises(ValueError, match="Invalid publication acknowledgement"):
            acknowledge(3)
        acknowledge(1)
        acknowledge(3)
    finally:
        window.close()


def test_slowest_consumer_holds_capacity_until_it_acknowledges(publication_window_type):
    window = publication_window_type(1, consumers=("fast", "slow"))
    try:
        fast = window.consumer("fast")
        slow = window.consumer("slow")
        fast(1)
        with pytest.raises(ValueError, match="Invalid publication acknowledgement"):
            fast(2)
        slow(1)
        fast(2)
        slow(2)
    finally:
        window.close()


def test_capacity_one_rollover_and_duplicate_acknowledgement(publication_window_type):
    window = publication_window_type(1)
    try:
        for batch_id in (1, 2, 3, 4):
            window.release(batch_id)
        window.consumer("default")(1)
    finally:
        window.close()


def test_invalid_acknowledgements_and_consumer_names(publication_window_type):
    window = publication_window_type(1)
    try:
        acknowledge = window.consumer("default")
        for batch_id in (0, 2):
            with pytest.raises(ValueError, match="Invalid publication acknowledgement"):
                acknowledge(batch_id)
        with pytest.raises(ValueError, match="Invalid publication acknowledgement"):
            window.consumer("unknown")
    finally:
        window.close()

    with pytest.raises(ValueError, match="Publication consumers must have unique names"):
        publication_window_type(1, consumers=("worker", "worker"))
    with pytest.raises(RuntimeError, match="Invalid batch publication window"):
        publication_window_type(0)


def test_callback_keeps_owner_alive_and_close_is_idempotent(publication_window_type):
    window = publication_window_type(1)
    acknowledge = window.consumer("default")
    reference = weakref.ref(window)
    del window
    gc.collect()
    assert reference() is not None
    acknowledge(1)
    reference().close()
    reference().close()
    with pytest.raises(RuntimeError, match="Publication window is closed"):
        acknowledge(1)


def test_sixty_four_consumers_can_complete_one_batch(publication_window_type):
    names = tuple(f"consumer-{index}" for index in range(64))
    window = publication_window_type(1, consumers=names)
    try:
        callbacks = [window.consumer(name) for name in names]
        for callback in callbacks:
            callback(1)
    finally:
        window.close()
