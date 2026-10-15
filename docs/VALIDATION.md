# Validation performed on 15 October 2026

**17 distinct tests passed locally**, in separate dependency environments. These are implementation tests with synthetic data, not research benchmark reproductions.

| Test group | Result | What was exercised |
|---|---|---|
| test_core.py | 9 passed | Capture/scene/exact-pixel leakage, path containment, shape contract, known MSE/PSNR, imbalance, absent classes and capture bootstrap |
| test_pipeline.py | 6 passed | RGB/BGR, resizing, exposure fusion, motion rejection/reset, thermal units, 1x/2x/4x forward shapes and gradients, two training epochs, checkpoint reload, evaluation and benchmark |
| test_export.py | 1 passed | ONNX export of fixed 1×3×8×8 input; max absolute PyTorch/ONNX difference <= 1e-5 on a random fixture |
| test_tensorflow.py | 1 passed | Synthetic TensorFlow convolution SavedModel, representative calibration, int8 input/output, dequantized output absolute error < 0.03 on a fixed fixture |

The pipeline group was rerun after adding the identity/bilinear evaluation baseline and experiment hashes. Python compileall passed. ONNX initially failed because optional progress output included a checkmark that the Windows cp932 console could not encode; the export call now explicitly disables verbose progress and its parity test passed. This is an observed local failure and repair, not a claim about an unverified upstream issue.

## Environment

Windows, Python 3.12.14, CPU:
- PyTorch 2.14.1+cpu
- NumPy 2.5.4
- OpenCV runtime 5.0.0 (opencv-python-headless wheel 5.0.0.93)
- ONNX 1.23.2, onnxscript 0.7.2, ONNX Runtime 1.31.0
- TensorFlow 2.21.0 in a separate installation directory

Top-level tested versions are recorded in requirements-tested.txt and requirements-tensorflow.txt. These are not complete transitive dependency locks. The shell used workspace-local package directories via PYTHONPATH; ordinary users should use separate virtual environments.

TensorFlow conversion passed. Its **test-only** tf.lite.Interpreter emitted a deprecation notice referring users to the LiteRT interpreter. The production conversion helper uses the official TFLiteConverter API. Do not interpret this test as certifying future interpreter availability.

## Repeat the checks

In a fresh Python 3.12 environment:

```sh
python -m pip install -r requirements-tested.txt
python -m unittest discover -s tests -p test_core.py -v
python -m unittest discover -s tests -p test_pipeline.py -v
python -m unittest discover -s tests -p test_export.py -v
```

In a separate environment, install requirements-tensorflow.txt and run:

```sh
python -m unittest discover -s tests -p test_tensorflow.py -v
```

Optional test groups deliberately skip if their dependencies are missing. A skip is not a pass; verify the summary and package availability.

## What has not been established

No real camera dataset, held-out sensor study, research checkpoint reproduction, CUDA memory soak, long video benchmark, 4K target latency, temperature accuracy, power consumption, vendor compiler result, FPGA/ASIC implementation, patent clearance or legal certification was performed.

No numerical model ranking or claim that all bias/overfitting/leaks are eliminated is made. The publication includes original code, tests and cited analysis only. Real deployments still need measured acceptance criteria and authorised data.
