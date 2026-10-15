import importlib.util
import tempfile
import unittest
from pathlib import Path
import numpy as np

@unittest.skipUnless(importlib.util.find_spec("tensorflow"), "Optional TensorFlow not installed")
class TensorFlowTests(unittest.TestCase):
    def test_integer_conversion(self):
        import tensorflow as tf
        from cdsp.deploy import integer_tflite
        class Fixture(tf.Module):
            @tf.function(input_signature=[tf.TensorSpec([1,8,8,1], tf.float32, name="image")])
            def serve(self, image):
                return {"output": tf.nn.relu(tf.nn.conv2d(image, tf.ones([3,3,1,1])/9, 1, "SAME"))}
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            fixture = Fixture()
            tf.saved_model.save(fixture, str(root/"saved"), signatures=fixture.serve)
            paths = []
            for i in range(5):
                path = root/f"calibration-{i}.npy"
                np.save(path, np.random.default_rng(i).random((1,8,8,1), dtype=np.float32))
                paths.append(path)
            result = root/"fixture.tflite"
            self.assertGreater(integer_tflite(root/"saved", paths, result), 0)
            interpreter = tf.lite.Interpreter(model_path=str(result))
            interpreter.allocate_tensors()
            inp = interpreter.get_input_details()[0]
            out = interpreter.get_output_details()[0]
            self.assertEqual(inp["dtype"], np.int8)
            self.assertEqual(out["dtype"], np.int8)
            x = np.full((1,8,8,1), .5, np.float32)
            scale, zero = inp["quantization"]
            quantized = np.clip(np.round(x/scale+zero), -128, 127).astype(np.int8)
            interpreter.set_tensor(inp["index"], quantized)
            interpreter.invoke()
            sy, zy = out["quantization"]
            actual = (interpreter.get_tensor(out["index"]).astype(np.float32)-zy)*sy
            expected = fixture.serve(tf.constant(x))["output"].numpy()
            self.assertLess(np.max(np.abs(actual-expected)), .03)
