import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from cdsp.classical import denoise, resize, exposure_fusion, TemporalDenoiser, thermal_correct
from cdsp.model import TinyRestorer
from cdsp.synthetic import generate
from cdsp.experiment import train, test, benchmark, load_model

class PipelineTests(unittest.TestCase):
    def test_colour_and_resize(self):
        red = np.zeros((16,16,3), dtype=np.uint8)
        red[:,:,0] = 200
        self.assertEqual(resize(red).shape, (32,32,3))
        result = denoise(red)
        self.assertGreater(result[:,:,0].mean(), result[:,:,2].mean()+100)

    def test_temporal_reset_and_motion(self):
        t = TemporalDenoiser()
        black = np.zeros((16,16,3), np.uint8)
        white = np.full_like(black, 255)
        t(black)
        np.testing.assert_array_equal(t(white), np.ones_like(white))
        t.reset()
        np.testing.assert_array_equal(t(black), black)

    def test_exposure(self):
        a = np.tile(np.arange(16, dtype=np.uint8)[None,:,None], (16,1,3))*10
        r = exposure_fusion([a, np.minimum(a.astype(int)*2,255).astype(np.uint8)])
        self.assertEqual(r.shape, a.shape)
        self.assertTrue(np.isfinite(r).all())
        self.assertTrue((r>=0).all() and (r<=1).all())

    def test_thermal_units(self):
        raw = np.full((4,4), 1000, dtype=np.uint16)
        np.testing.assert_array_equal(thermal_correct(raw, np.full((4,4), .1), np.full((4,4), -50)), np.full((4,4),50))
        with self.assertRaises(ValueError):
            thermal_correct(raw, np.zeros((4,4)), np.zeros((4,4)))

    def test_shapes_and_gradients(self):
        for scale in (1,2,4):
            m = TinyRestorer(1,4,scale)
            out = m(torch.rand(1,1,8,8))
            self.assertEqual(tuple(out.shape), (1,1,8*scale,8*scale))
            out.mean().backward()
            self.assertTrue(all(p.grad is not None for p in m.parameters()))

    def test_training_and_evaluation(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as d:
            manifest = generate(Path(d)/"data")
            checkpoint = Path(d)/"model.pt"
            report = train(manifest, checkpoint, epochs=2)
            self.assertFalse(report["test_used_for_selection"])
            r = test(manifest, checkpoint, Path(d)/"result.json")
            self.assertTrue(r["summary"]["complete"])
            model, _ = load_model(checkpoint)
            with torch.inference_mode():
                self.assertFalse(model(torch.zeros(1,3,8,8)).requires_grad)
            self.assertGreater(benchmark(model, (1,3,8,8), repeats=2)["median_ms"], 0)
