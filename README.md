# CDSP: a small camera-algorithm laboratory

Imagine a camera is a little painter. Sometimes its picture has snow, looks too small, or has a very bright window. This project gives the painter small tools. We also give it a fair exam: it must not practise on the exam pictures.

This is an independent educational project for the PD0150 skill areas. It is not an employer product, an implementation of its confidential CDSP, a reproduced research leaderboard, or a finished camera chip. “Interperation” in the advert is treated as image interpolation; the actual intended term is unconfirmed.

## Try it

Use Python 3.12 in a new environment:

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m cdsp.synthetic --output data --scale 1
python -m cdsp.experiment train --manifest data/manifest.json --checkpoint runs/model.pt --epochs 5
python -m cdsp.experiment test --manifest data/manifest.json --checkpoint runs/model.pt --output runs/evaluation.json
python -m cdsp.experiment benchmark --checkpoint runs/model.pt
```

For a 2x SR plumbing test, generate a different directory with `--scale 2`, train with `--scale 2`, and use a different checkpoint. Synthetic random tiles prove that the pipes connect; they do not prove good photographs.

## What is implemented

| Job skill | Working entry point | Plain explanation and boundary |
|---|---|---|
| Spatial denoising | `classical.denoise` | Smooth visible-image noise with OpenCV NLM. |
| TNR | `TemporalDenoiser` | Blend only similar pixels in two frames. Reset each capture. Motion compensation is not implemented. |
| WDR | `exposure_fusion` | Combine already aligned exposures for display. This does not recover calibrated radiance. |
| Interpolation | `classical.resize` | Make more pixels with bicubic interpolation. |
| AI denoising / SR | `TinyRestorer`, train/test CLI | A small original residual CNN with scale 1, 2 or 4. No claim of paper-level quality. |
| Thermal | `thermal_correct` | Apply supplied gain/offset maps, preserving units. This does not estimate temperature or calibration maps. |
| Model compiler | `deploy.export_onnx`, `compare_onnx` | Export and check numerical agreement. No vendor compiler, RTL or timing closure is supplied. |
| TensorFlow integer deployment | `deploy.integer_tflite` | Convert a separately supplied TensorFlow SavedModel with representative training data; not a PyTorch-to-TensorFlow bridge. |
| Research / technical documents | [Research](docs/RESEARCH.md), [three causes](docs/DIAGNOSIS.md), [complexity](docs/COMPLEXITY.md) | Compare primary sources and explain the limits. No novelty or patentability claim. |

All production functions include a complexity explanation, including helper functions. Source mappings and date rules are in [SOURCES](docs/SOURCES.md). [VALIDATION](docs/VALIDATION.md) records what actually ran.

## Use real data carefully

Create a JSON list following `cdsp.synthetic.generate`: id, split, capture, scene, sensor, stratum, input and target. Arrays are float32 CHW in [0,1], RGB or one channel; targets have the chosen integer spatial scale. Paths are relative to the manifest directory. Group all related frames, crops, exposures and augmentations under their original capture and scene **before splitting**. Names must be globally consistent.

The validator hashes decoded arrays and rejects cross-split exact duplicates, capture overlap and scene overlap. It cannot discover renamed near-duplicates or false metadata. Audit those manually; keep a separate unseen-sensor/site/date test set. Pretrained model training sets may be unknown, so their contamination cannot be ruled out.

Predeclare strata such as sensor × illumination × motion × temperature band. This code reports frame-weighted MSE, equal-capture/equal-stratum MSE, worst stratum, counts and a separate equal-capture bootstrap interval. Missing required strata makes the aggregate incomplete. A same-input identity/bilinear baseline is also reported. PSNR uses range 1, all channels, the full image and no prediction clipping; perfect PSNR is JSON null plus `perfect=true`. Never compare this directly with papers using Y-only scoring or shaved borders.

Training uses inverse capture/stratum weights and validation-only early stopping. Test data is not used to choose weights, epochs or parameters. Repeatedly looking at the test set can still overfit it: freeze the protocol and run a final held-out exam once. Classification metrics are optional for a downstream classifier; “accuracy” is not an image-restoration metric.

## Deployment example

```sh
python -m pip install -r requirements-export.txt
python -c "from cdsp.deploy import export_onnx; export_onnx('runs/model.pt', 'runs/model.onnx')"
```

The exported shape is fixed at 1×C×64×64 unless size is changed. Run `compare_onnx` with that exact shape. Quantization needs an independent accuracy check, compiler operator coverage, saturation tests, memory and latency measurements on the actual chip. TensorFlow is optional: install requirements-tensorflow.txt in a separate environment; use arrays matching the SavedModel input signature.

No CUDA, 4K, FPGA or ASIC performance is asserted. One-image streaming bounds memory but may be slower than a tuned batch loader. FP32 is the reference; enabling AMP, compilation or quantization requires a fresh parity/quality measurement.

## Publication and Taiwan

Only original source, synthetic test logic and short cited analysis are included. No job-advert copy, employer assets, captured personal images, paper PDFs, upstream weights or research implementation code is redistributed. See [publication notes](docs/LEGAL.md). This limited publication plan is not a legal guarantee or patent clearance.
