# Technology selection, checked 15 October 2026

The three-year paper window is **15 October 2023–15 October 2026**, using conference presentation dates, not the preprint upload. The diagnostic window is **15 October 2021–15 October 2026**. CVPR 2023 and early-October ICCV 2023 are outside the first window. Workshop papers are not silently treated as main-conference papers.

This is a curated comparison, not an exhaustive global “most starred” search. These five author repositories were checked for relevant accepted papers, public implementation, model links and reproduction instructions. Stars below are GitHub's displayed snapshot (rounded where shown), not a reproducibility score. Independent reproduction has **not** been completed.

| Popularity within this shortlist | Accepted work / author implementation | Fit for the job | Limitation / decision |
|---|---|---|---|
| 1: 5.7k | [SUPIR, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Yu_Scaling_Up_to_Excellence_Practicing_Model_Scaling_for_Photo-Realistic_Image_CVPR_2024_paper.html), [code](https://github.com/Fanghua-Yu/SUPIR) | Generative photo restoration; public inference and model instructions | Research comparison only. [Commercial licensing is separate](https://github.com/Fanghua-Yu/SUPIR/blob/master/LICENSE). No measured embedded suitability. |
| 2: 4.1k | [DiffBIR, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/7690_ECCV_2024_paper.php), [code](https://github.com/XPixelGroup/DiffBIR) | Blind restoration with generative diffusion prior | Compare visual plausibility and measurement fidelity separately. Current repository versions are not automatically the conference checkpoint. |
| 3: 1.2k | [MambaIR, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/2740_ECCV_2024_paper.php), [code](https://github.com/csguoh/MambaIR) | Restoration/SR with state-space modelling and local enhancement | First research candidate for a fidelity-oriented study. Verify selective-scan operator support on the target compiler. Stars are shared with later MambaIRv2. |
| 4: 658 | [SeeSR, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Wu_SeeSR_Towards_Semantics-Aware_Real-World_Image_Super-Resolution_CVPR_2024_paper.pdf), [code](https://github.com/cswry/SeeSR) | Semantic conditioning for real-world SR | Useful perceptual comparator; semantic plausibility does not establish that the captured scene contained the generated texture. |
| 5: 546 | [PromptIR, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/e187897ed7780a579a0d76fd4a35d107-Abstract-Conference.html), [code](https://github.com/va1shn9v/PromptIR) | All-in-one blind restoration with learned prompts | Research comparator. [Academic licence excludes commercial use](https://github.com/va1shn9v/PromptIR/blob/main/LICENSE.md). |

The fit and deployment decisions are engineering judgements, not measured superiority claims. No cross-paper PSNR table is reproduced: degradation, data splits, channels, crops, weights and inference budgets differ. None of these visible-image results establishes thermal radiometry, temporal stability, RAW Bayer processing, WDR performance or silicon timing.

**Practical selection:** use the provided classical tools and tiny CNN to validate the evaluation/deployment path first; then test the appropriate official MambaIR variant on the same licensed, held-out data. Retain generative models as separately scored perceptual candidates. The tiny CNN is an original composition of documented framework primitives, not a new state-of-the-art method or a reproduction of any listed paper.

## Framework choice

| Tool | Role | Reason and boundary |
|---|---|---|
| PyTorch | Reference training, inference, ONNX export | Official Conv2d, PixelShuffle, interpolation, AdamW and inference APIs. All shortlisted research implementations use the PyTorch ecosystem; pin each upstream environment separately. |
| OpenCV | Conventional visible-image reference | Official NLM, resize and MergeMertens APIs. RGB/BGR conversion is explicit. |
| TensorFlow / LiteRT | Optional integer conversion for an existing SavedModel | Official full-integer conversion with representative data; no unverified automatic conversion between framework models. |
| Vendor compiler, FPGA / ASIC tools | Later target validation | Not specified in the advert; instruction set, supported operators, SRAM, bit widths and clock constraints remain unknown. |

Before comparing official models: record repository commit, checkpoint hash, data rights and dataset hashes; use the author's matching config and dependencies; reproduce one stated benchmark protocol; then run a new shared protocol without retuning on test data. No upstream code or weights are bundled here.
