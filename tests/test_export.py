import importlib.util
import tempfile
import unittest
from pathlib import Path
import numpy as np

@unittest.skipUnless(all(importlib.util.find_spec(p) for p in ("onnx","onnxscript","onnxruntime")), "Optional ONNX dependencies not installed")
class ExportTests(unittest.TestCase):
    def test_parity(self):
        import torch
        from cdsp.model import TinyRestorer
        from cdsp.deploy import export_onnx, compare_onnx
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            torch.manual_seed(5)
            m = TinyRestorer()
            torch.save({"state_dict": m.state_dict(), "channels":3, "width":16, "scale":1}, p/"model.pt")
            export_onnx(p/"model.pt", p/"model.onnx", size=8)
            x = np.random.default_rng(2).random((1,3,8,8), dtype=np.float32)
            self.assertLessEqual(compare_onnx(p/"model.pt", p/"model.onnx", x)["max_absolute_error"], 1e-5)
