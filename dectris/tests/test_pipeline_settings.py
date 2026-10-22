"""Defaults and validation of the pipeline example's YAML settings (no holoscan)."""

import os
import sys
import unittest

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples",
                             "pipeline"))

from settings import DEFAULTS, resolve  # noqa: E402

CONNECTION = {"remote_address": "10.0.0.1:31003", "local_address": "10.0.0.2"}


class ResolveTest(unittest.TestCase):
    def test_defaults(self):
        s = resolve({"connection": CONNECTION})
        self.assertEqual(s["connection"]["ring_mib"], 64)
        self.assertEqual(s["connection"]["ring_gpu"], 0)
        self.assertEqual(s["execute"], {"seconds": 10.0, "threads": 3})
        self.assertEqual(s["receiver"], {"receive_only": False, "keep_images": False, "max_batch": 512})
        self.assertEqual(s["aggregate"], {"series_width": 0})

    def test_override(self):
        s = resolve({"connection": CONNECTION, "receiver": {"receive_only": True}})
        self.assertTrue(s["receiver"]["receive_only"])
        self.assertFalse(s["receiver"]["keep_images"])

    def test_required(self):
        with self.assertRaisesRegex(ValueError, "local_address"):
            resolve({"connection": {"remote_address": "a:1"}})
        with self.assertRaisesRegex(ValueError, "remote_address"):
            resolve({})

    def test_unknown(self):
        with self.assertRaisesRegex(ValueError, "section"):
            resolve({"connection": CONNECTION, "bogus": {}})
        with self.assertRaisesRegex(ValueError, "gpuu"):
            resolve({"connection": {**CONNECTION, "gpuu": 1}})

    def test_shipped_yaml_keys_are_known(self):
        import yaml
        path = os.path.join(os.path.dirname(__file__), "..", "examples", "pipeline",
                            "pipeline.yaml")
        shipped = yaml.safe_load(open(path))
        self.assertEqual(set(shipped), set(DEFAULTS))
        resolve(shipped)
        for name, keys in shipped.items():
            self.assertLessEqual(set(keys), set(DEFAULTS[name]))


if __name__ == "__main__":
    unittest.main()
