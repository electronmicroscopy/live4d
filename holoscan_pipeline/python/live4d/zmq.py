"""Acquisition admission and result ownership around one ZMQ source."""
from .operators import _PublicationWindow
from .results import ResultSlot


class ZmqReceiver:
    """One outstanding acquisition per routing lane, with source backpressure."""

    def __init__(self, config):
        self.results = []
        self.acquisitions = []
        self.inputs = []
        self.finished = [0] * len(config.gpu_devices)
        self.completed = 0
        self.finite = config.execute["mode"] == "finite"
        try:
            for device in config.gpu_devices:
                self.results.append(ResultSlot(device))
                self.acquisitions.append(_PublicationWindow(1))
                self.inputs.append(_PublicationWindow(config.queue_depth - 1))
        except BaseException:
            self.close()
            raise

    def completion(self, index):
        def submit(totals, finish):
            slot = self.results[index]
            slot.available.clear()

            def completed(values):
                finish(values)
                self.finished[index] += 1

            slot.submit(totals, completed)
        return submit

    def tick(self):
        for index, slot in enumerate(self.results):
            if slot.poll():
                # Reopen ingress only after poll releases the old slot reservation.
                if not self.finite:
                    self.acquisitions[index].release(self.finished[index])
                self.completed += 1

    def close(self):
        # Runtime must join before buffers and callbacks can be released.
        for slot in self.results:
            slot.close()
