"""Official export APIs only. Export is not proof of ASIC/FPGA support."""
from pathlib import Path
import numpy as np

def export_onnx(checkpoint, destination, size=64):
    """O(F) graph capture plus compiler passes (backend dependent), O(A+W) typical memory."""
    import torch
    from .experiment import load_model
    if size < 1:
        raise ValueError("Positive size required")
    model, config = load_model(checkpoint)
    sample = torch.zeros(1, config["channels"], size, size)
    torch.onnx.export(model, (sample,), str(destination), dynamo=True, verbose=False,
                      input_names=["image"], output_names=["restored"])
    return Path(destination)

def compare_onnx(checkpoint, onnx_path, image, atol=1e-5):
    """O(F_torch+F_onnx) time, O(A_torch+A_onnx) memory; exact fixed export shape required."""
    import torch
    import onnxruntime as ort
    from .experiment import load_model
    if image.dtype != np.float32 or not np.isfinite(image).all() or atol <= 0:
        raise ValueError("Finite float32 input and positive tolerance required")
    model, _ = load_model(checkpoint)
    with torch.inference_mode():
        reference = model(torch.from_numpy(image)).numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    result = session.run(None, {"image": image})[0]
    if result.shape != reference.shape or not np.isfinite(result).all():
        raise ValueError("Invalid exported output")
    error = float(np.max(np.abs(result-reference)))
    if error > atol:
        raise ValueError(f"Export parity failed: {error} > {atol}")
    return {"max_absolute_error": error, "atol": atol}

def integer_tflite(saved_model, calibration_arrays, destination):
    """O(M*F + compiler passes), O(A+W); independent TensorFlow SavedModel input."""
    import tensorflow as tf
    paths = list(calibration_arrays)
    if not paths:
        raise ValueError("Training-only representative calibration arrays required")
    def representative():
        """O(M*P) loading, O(P) memory; SavedModel input signature must match arrays."""
        for path in paths:
            a = np.load(path, allow_pickle=False)
            if a.dtype != np.float32 or not np.isfinite(a).all():
                raise ValueError("Finite float32 calibration required")
            yield [a]
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    result = converter.convert()
    Path(destination).write_bytes(result)
    return len(result)
