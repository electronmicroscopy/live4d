"""Start message of the imagemux bstream output -> series geometry, and the raster scan positions
of image ids (no RDMA, no GPU)."""

import datetime
import os
import sys
import unittest

import cbor2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from dectris.messages import ScanShape, Series, parse_start, scan_positions  # noqa: E402


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
        self.assertEqual(series.shape, (96, 96))

    def test_unbinned_series_shape(self):
        self.assertEqual(parse_start(start_message(image_size_x=192, image_size_y=192)).shape,
                         (192, 192))

    def test_announced_size_is_only_a_hint(self):
        self.assertEqual(parse_start(start_message(series_id=8, number_of_images=100)),
                         Series(8, 100))
        self.assertEqual(parse_start(start_message(number_of_images=0)), Series(4, None))
        self.assertEqual(parse_start(start_message(number_of_images="many")), Series(4, None))

    def test_announced_frame_time(self):
        self.assertEqual(parse_start(start_message(frame_time=5.58e-5)), Series(4, 1000, 5.58e-5))
        self.assertIsNone(parse_start(start_message(frame_time=0.0)).frame_time)
        self.assertIsNone(parse_start(start_message(frame_time="fast")).frame_time)

    def test_announced_scan(self):
        appendix = {"scan_width": 256, "scan_height": 128, "user": "x"}
        series = parse_start(start_message(header_appendix=appendix))
        self.assertEqual(series.scan, ScanShape(256, 128))
        self.assertEqual(series.shape, (96, 96))
        self.assertIsNone(parse_start(start_message()).scan)
        self.assertIsNone(parse_start(start_message(header_appendix={"user": "x"})).scan)
        self.assertIsNone(parse_start(start_message(header_appendix='{"scan_width": 4}')).scan)

    def test_rejects_invalid_announced_scan(self):
        for appendix in [{"scan_width": 256}, {"scan_height": 128},
                         {"scan_width": 0, "scan_height": 128},
                         {"scan_width": "256", "scan_height": 128},
                         {"scan_width": 256.0, "scan_height": 128},
                         {"scan_width": True, "scan_height": 128},
                         {"scan_width": 2 ** 31, "scan_height": 1}]:
            with self.assertRaises(ValueError):
                parse_start(start_message(header_appendix=appendix))

    def test_accepts_bytes_like(self):
        parse_start(memoryview(start_message()))

    def test_rejects(self):
        for message in [
            start_message(image_size_x=128),
            start_message(image_size_x=192, image_size_y=96),
            start_message(image_size_x=192, image_size_y=192, image_dtype="uint32"),
            start_message(image_dtype="uint32"),
            cbor2.dumps({"type": "end", "series_id": 4}),
            cbor2.dumps({"type": "start"}),
            b"\xff\xff",
        ]:
            with self.assertRaises(ValueError):
                parse_start(message)


class ScanTest(unittest.TestCase):
    def test_shape(self):
        self.assertEqual(ScanShape(1000, 1000).size, 1_000_000)
        for width, height in ((0, 1), (1, -1), (1.5, 2), (True, 2)):
            with self.assertRaises(ValueError):
                ScanShape(width, height)

    def test_raster_positions(self):
        positions, valid = scan_positions(np.array([0, 1, 4, 5, 11], dtype=np.uint64),
                                          ScanShape(4, 3))
        self.assertTrue(valid.all())
        self.assertEqual(positions.dtype, np.int32)
        np.testing.assert_array_equal(positions, [[0, 0], [0, 1], [1, 0], [1, 1], [2, 3]])

    def test_gaps_keep_their_place(self):
        positions, _ = scan_positions([2, 3, 7], ScanShape(4, 3))
        np.testing.assert_array_equal(positions, [[0, 2], [0, 3], [1, 3]])

    def test_ids_outside_the_scan(self):
        positions, valid = scan_positions([10, 11, 12, 13], ScanShape(4, 3))
        np.testing.assert_array_equal(valid, [True, True, False, False])
        np.testing.assert_array_equal(positions, [[2, 2], [2, 3]])
        positions, valid = scan_positions([], ScanShape(4, 3))
        self.assertEqual(positions.shape, (0, 2))


if __name__ == "__main__":
    unittest.main()
