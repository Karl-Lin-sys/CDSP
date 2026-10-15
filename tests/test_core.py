import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from cdsp.data import read_manifest, load_pair
from cdsp.metrics import errors, summarize, classification_counts, capture_bootstrap
from cdsp.synthetic import generate

class CoreTests(unittest.TestCase):
    def test_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            p = generate(d)
            rows = read_manifest(p)
            self.assertEqual(len(rows), 18)
            self.assertEqual(load_pair(rows[0], 1)[0].shape, (3, 16, 16))

    def test_capture_leakage(self):
        with tempfile.TemporaryDirectory() as d:
            p = generate(d)
            rows = json.loads(p.read_text())
            rows[6]["capture"] = rows[0]["capture"]
            p.write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError, "leakage"):
                read_manifest(p)

    def test_duplicate_pixels_even_when_renamed(self):
        with tempfile.TemporaryDirectory() as d:
            p = generate(d)
            rows = json.loads(p.read_text())
            np.save(Path(d)/rows[6]["input"], np.load(Path(d)/rows[0]["target"]))
            with self.assertRaisesRegex(ValueError, "leakage"):
                read_manifest(p)

    def test_scene_leakage(self):
        with tempfile.TemporaryDirectory() as d:
            p = generate(d)
            rows = json.loads(p.read_text())
            rows[6]["scene"] = rows[0]["scene"]
            p.write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError, "leakage"):
                read_manifest(p)

    def test_path_escape(self):
        with tempfile.TemporaryDirectory() as d:
            p = generate(d)
            rows = json.loads(p.read_text())
            rows[0]["input"] = "../outside.npy"
            p.write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError, "inside"):
                read_manifest(p)

    def test_metrics(self):
        a = np.zeros((1, 2, 2), dtype=np.float32)
        b = np.ones_like(a)*0.5
        self.assertAlmostEqual(errors(a, b)["mse"], 0.25)
        self.assertAlmostEqual(errors(a, b)["psnr_db"], 6.020599913)
        self.assertIsNone(errors(a, a)["psnr_db"])
        with self.assertRaises(ValueError):
            errors(a*np.nan, b)

    def test_macro_resists_frame_imbalance(self):
        records = [{"stratum": "easy", "capture": "a", "mse": 0.01}]*99
        records += [{"stratum": "hard", "capture": "b", "mse": 0.25}]
        result = summarize(records, ["easy", "hard"])
        self.assertAlmostEqual(result["macro_mse"], 0.13)
        self.assertLess(result["frame_micro_mse"], 0.02)
        self.assertFalse(summarize(records, ["easy", "hard", "absent"])["complete"])

    def test_absent_class_undefined(self):
        r = classification_counts([[9, 1], [0, 0]])
        self.assertIsNone(r["balanced_accuracy"])
        self.assertIsNone(r["recall"][1])
        self.assertAlmostEqual(classification_counts([[9, 1], [5, 5]])["balanced_accuracy"], .7)

    def test_bootstrap_unit(self):
        rows = [{"capture": "a", "mse": .1}, {"capture": "b", "mse": .3}]
        self.assertEqual(capture_bootstrap(rows, 100), capture_bootstrap(rows, 100))
        self.assertIsNone(capture_bootstrap(rows[:1])["ci95"])
