# Three testable causes and concrete remedies

No defective image, training run, sensor calibration or compiler log was supplied. Therefore **there is no identified root cause for this employer's system**. The following are falsifiable mechanisms and implemented controls, not accusations or guesses presented as findings.

## 1. The exam contains practice pictures: leakage and model-selection overfitting

**Mechanism:** if two crops or neighbouring frames share the same scene, a split by filename can put related information in both training and testing. The test no longer measures unseen-scene performance. Repeatedly choosing a model after reading test scores also makes the test part of selection. Leakage and misleading evaluation are documented by [Kapoor and Narayanan, Patterns 2023](https://doi.org/10.1016/j.patter.2023.100804), a peer-reviewed diagnostic source within five years, not part of the three-year architecture shortlist.

**Confirm/reject:** audit source capture/scene identities and decoded-image hashes; compare the locked model on an independently captured, unseen-sensor/site/date set. A lower score alone does not prove leakage: domain shift also changes difficulty.

**Remedy implemented:** reject cross-split capture, scene and exact-pixel overlap; select checkpoints and early stopping only on validation. Preserve test data for a final frozen evaluation. The validator deliberately rejects an exact duplicate even when renamed or used as an input in one split and target in another.

**Limit:** hashes miss resized/recompressed duplicates, metadata can be wrong, and pretrained training data may be unavailable. No code can certify all leakage absent. Human provenance review and near-duplicate inspection remain necessary.

Little explanation: keep the practice box and the exam box apart.

## 2. Many easy pictures hide a few hard ones: selection bias and metric imbalance

**Mechanism, derived here:** a frame-weighted risk is R = sum_g p_g R_g. With 99 easy frames at MSE 0.01 and one hard frame at 0.25, R = 0.0124. Equal-condition risk is (0.01 + 0.25)/2 = 0.13. Both numbers are correct for different questions. Neither can describe a condition that was never sampled. The same diagnostic paper above discusses misleading metric choice under class imbalance; the restoration weighting here is an explicit engineering adaptation.

**Confirm/reject:** tabulate independent capture counts for sensor, scene, light, motion and temperature bands. Report both deployment-frequency and equal-condition results. Collect missing bands rather than manufacturing confidence through oversampling the same capture.

**Remedy implemented:** training uses inverse stratum/capture/frame-count weights; evaluation averages frames within capture, captures within stratum, then strata equally. Missing strata block complete aggregate scores. Bootstrap samples whole captures and labels its different equal-capture estimand. Optional classifier metrics return undefined macro scores when a true class is absent.

**Limit:** reweighting cannot repair missing support, wrong labels or a biased capture process. Rare noisy groups can gain influence. Define deployment weights before opening test results; this implementation does not claim a universal unbiased population estimate.

Little explanation: give each kind of picture a turn.

## 3. A pretty picture can be an inaccurate measurement: structural and deployment mismatch

[DiffBIR (ECCV 2024)](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/7690_ECCV_2024_paper.php) uses generative restoration; [SeeSR (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/papers/Wu_SeeSR_Towards_Semantics-Aware_Real-World_Image_Super-Resolution_CVPR_2024_paper.pdf) uses semantic guidance. Their task design does not establish calibrated thermal accuracy. [MambaIR (ECCV 2024)](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/2740_ECCV_2024_paper.php) addresses local forgetting/channel redundancy in a state-space restoration backbone; this does not prove compatibility with a particular model compiler.

**Mechanism, derived here:** downsampling y = D x is many-to-one. If D v = 0, both x and x+v explain y. A prior can choose plausible detail but cannot identify unobserved truth from y alone. Likewise, two-frame blending across unmodelled motion can create ghosting. Quantized rounding with step delta gives at most delta/2 scalar error before saturation; a network's output error can amplify through subsequent operators. Clipping breaks that bound.

**Confirm/reject:** test known patterns, moving edges, scene cuts and calibrated thermal targets; compare pre/post-export outputs on the same tensors; measure saturation, operator fallbacks, peak memory and latency on the actual device. No claim of these failures is made without these measurements.

**Remedy implemented:** keep thermal gain/offset correction in original units, separate from display fusion; reject temporal blending for changed pixels and reset at capture boundaries; use a small explicit CNN as a deployment control; export ONNX and assert numerical parity. The optional [official TensorFlow integer route](https://developers.google.com/edge/litert/conversion/tensorflow/quantization/post_training_integer_quant) restricts operators to integer built-ins, failing conversion rather than knowingly allowing float fallback.

**Limit:** temporal gating is not flow alignment; calibration maps must come from real measurements; full integer conversion is not hardware timing proof. The tiny network has a limited receptive field and no temporal input. Neither patent novelty nor compiler support can be established from a job advert.

Little explanation: a drawing that looks nice is not always a true ruler.
