"""Start message of the imagemux bstream output -> series geometry (no RDMA, no GPU)."""

import datetime
import os
import sys
import unittest

import cbor2

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from dectris.messages import Series, parse_start  # noqa: E402


def start_message(**overrides) -> bytes:
    """The start message a DAQ simulator sent (rdma_dump --decode), minus fields we ignore."""
    msg = {
        "type": "start",
        "series_id": 4,
        "series_unique_id": "01M5AJXTNHFFNW2Q3A9K3YBZ14",
        "arm_date": datetime.datetime(2026, 10, 19, 17, 21, 40, tzinfo=datetime.timezone.utc),
        "channels": ["threshold_1"],
        "count_time": 8.234e-6,
        "countrate_correction_enabled": True,
        "detector_description": "Dectris ARINA Si",
        "image_dtype": "uint16",
        "image_size_x": 96,
        "image_size_y": 96,
        "number_of_images": 1000,
        "threshold_energy": {"threshold_1": 2000.0},
    }
    msg.update(overrides)
    return cbor2.dumps(cbor2.CBORTag(55799, msg))  # self-describe tag, as the real one has


class ParseStartTest(unittest.TestCase):
    def test_series(self):
        series = parse_start(start_message())
        self.assertEqual(series, Series(4, 1000))

    def test_announced_size_is_only_a_hint(self):
        self.assertEqual(parse_start(start_message(series_id=8, number_of_images=100)),
                         Series(8, 100))
        self.assertEqual(parse_start(start_message(number_of_images=0)), Series(4, None))
        self.assertEqual(parse_start(start_message(number_of_images="many")), Series(4, None))

    def test_announced_frame_time(self):
        self.assertEqual(parse_start(start_message(frame_time=5.58e-5)), Series(4, 1000, 5.58e-5))
        self.assertIsNone(parse_start(start_message(frame_time=0.0)).frame_time)
        self.assertIsNone(parse_start(start_message(frame_time="fast")).frame_time)

    def test_accepts_bytes_like(self):
        parse_start(memoryview(start_message()))

    def test_rejects(self):
        for message in [
            start_message(image_size_x=128),
            start_message(image_dtype="uint32"),
            cbor2.dumps({"type": "end", "series_id": 4}),
            cbor2.dumps({"type": "start"}),
            b"\xff\xff",
        ]:
            with self.assertRaises(ValueError):
                parse_start(message)


if __name__ == "__main__":
    unittest.main()
