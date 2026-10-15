# Primary-source register and code provenance

Access date: 15 October 2026, Asia/Taipei. Only accepted-paper proceedings, author repositories, official framework documentation and Taiwan's official legislation are used. Search results from blogs, forums, unofficial mirrors and under-review papers were not used as evidence.

The architecture comparison uses the five accepted papers and author repositories linked in [RESEARCH](RESEARCH.md). These satisfy the three-year conference window. The Patterns 2023 paper in [DIAGNOSIS](DIAGNOSIS.md) satisfies the separate five-year peer-reviewed diagnostic window; it is not labelled a top-conference architecture source. No unresolved issue or general troubleshooting article is cited.

## Official API mapping

| Code | Official reference |
|---|---|
| TinyRestorer convolutions | [torch.nn.Conv2d](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Conv2d.html) |
| Residual base resizing | [functional.interpolate](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.interpolate.html) |
| Output rearrangement | [PixelShuffle](https://docs.pytorch.org/docs/2.14/generated/torch.nn.PixelShuffle.html) |
| Gradient updates | [AdamW](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html), [zero_grad](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Optimizer.zero_grad.html) |
| Evaluation without graph retention | [inference_mode](https://docs.pytorch.org/docs/2.14/generated/torch.autograd.grad_mode.inference_mode.html) |
| Seeding and deterministic settings | [reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html) |
| Checkpoint loading | [torch.load](https://docs.pytorch.org/docs/2.14/generated/torch.load.html) |
| ONNX export | [torch.onnx](https://docs.pytorch.org/docs/2.14/onnx.html) |
| NLM denoising | [OpenCV denoising](https://docs.opencv.org/4.x/d1/d79/group__photo__denoise.html) |
| Bicubic resize | [OpenCV transformations](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html) |
| Exposure fusion | [OpenCV official HDR tutorial](https://docs.opencv.org/4.x/d2/df0/tutorial_py_hdr.html) |
| Full integer SavedModel conversion | [TensorFlow link, redirected to official LiteRT](https://developers.google.com/edge/litert/conversion/tensorflow/quantization/post_training_integer_quant) |

The PyTorch stable links resolved to 2.14 documentation. OpenCV's 4.x documentation resolved to 4.13.0; the locally available wheel version is recorded in VALIDATION. This difference is disclosed rather than equating a docs alias with an installed version. TensorFlow documentation is a live official page; conversion support must be checked against the installed package.

The wrappers, split audit, metrics, CNN composition, synthetic fixtures, tests and complexity derivations are original project code. They are **not** copied official research implementations. Standard array operations express arithmetic directly. The thermal affine map and temporal gate are transparent engineering controls, not claimed recent-paper algorithms. No official framework example is claimed to prove their camera quality.

Documentation sources are used for API contracts; papers are used for research claims. Mathematical risk weighting, inverse-problem reasoning and Big-O bounds are explicitly derived for this code. The two should not be confused.

## Reproducibility boundaries

Paper reproducibility is stronger than having stars, a README or a download link. The shortlisted repositories provide public implementation/model/evaluation materials, but no independent reproduction was performed here. Stars are current displayed rounded counts and may change; they are only a sorting signal within the disclosed shortlist.

Before running upstream research models, save immutable commit IDs, matching checkpoint hashes, licence versions and configuration files. This project downloads none automatically. Therefore no unverified upstream checkpoint provenance or benchmark claim is presented.

Latest framework behaviour was checked through the official pages above. No resolved issue was needed to justify a bug fix in those frameworks; this project does not invent an issue ID or attribute an unverified memory leak to a framework.
