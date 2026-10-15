# How the work grows

Big-O is a growth rule, not a stopwatch. Doubling both image sides gives four times as many pixels.

Notation: P=HWC input samples; N=number of frames; G=number of strata/capture groups; E=training epochs or exposure count where stated; s=upscale factor; D=hidden channels; B=batch size; R=bootstrap draws or benchmark repeats; K=classes; W=parameter count; A=one-pass activations; F=network forward cost. Fixed-size hashes are treated as constant-time dictionary keys; I/O includes reading decoded arrays. Bounds describe this implementation, not a proprietary OpenCV/driver allocation guarantee.

| Functions | Time | Additional memory / explanation |
|---|---|---|
| array_hash | O(P) | O(P); decoded contiguous pixels and byte copy |
| read_manifest | O(N + total pixels) | O(N + max P); metadata and one hash image at a time |
| load_pair, tensor_pair | O(s²P) | O(s²P); validation and one paired sample |
| errors | O(s²P) | O(s²P); float64 differences |
| summarize | O(N+G) | O(N+G); stores scalar errors, never GPU graphs |
| capture_bootstrap | O(RN) upper bound | O(N+R); one resample at a time |
| classification_counts | O(K²) | O(K²); square confusion matrix |
| rgb8 | O(1) | O(1); only shape/dtype checks (docstring upper bound O(P) is conservative) |
| denoise | O(P S² T²) conservative naive bound | O(P) for fixed search S=21, template T=7; OpenCV's fast kernel may do less work |
| resize | O(s²P) | O(s²P); fixed bicubic footprint |
| exposure_fusion | O(E P) | O(E P); E aligned exposures and geometric image pyramids |
| TemporalDenoiser.__init__, reset | O(1) | O(1), apart from reference release |
| TemporalDenoiser.__call__ | O(P) | O(P); only one prior raw frame persists |
| thermal_correct | O(P) | O(P); calibration arrays already supplied |
| TinyRestorer.__init__ | O(W) | O(W) |
| TinyRestorer.forward | O(BHW·9·(CD+D²+DCs²)) | O(BHW(D+Cs²)) inference activations; parameters separate |
| seed_all | O(1) | O(1) for fixed device count |
| evaluate | O(NF) | O(A+N), parameters separate |
| train | O(E N F) | O(A+W+N); forward/backward constant multiple, AdamW state O(W), best state O(W) |
| load_model | O(W) | O(W) |
| test | O(NF+RN) | O(A+N+R), plus parameters |
| benchmark | O((5+R)F) | O(A+R), plus parameters; five warmups |
| main | selected command | dispatch O(1) |
| generate | O(N s² P) | O(s²P+N), disk O(N s²P) |
| export_onnx | capture O(F), compiler dependent thereafter | typically O(A+W); no unsupported universal compiler bound |
| compare_onnx | O(F_torch+F_onnx) | O(A_torch+A_onnx+W) |
| integer_tflite | O(MF) calibration + compiler-dependent passes | O(A+W+M) paths, M calibration samples |
| representative (nested generator) | O(MP) data loading | O(P); one calibration input |

The network does its convolutions at input resolution, then rearranges channels with PixelShuffle. That keeps intermediate feature maps smaller than first upscaling every hidden feature.

Training runs FP32, clears gradients with set_to_none, and converts logged losses to plain scalars. Evaluation combines eval() with inference_mode(). No tensor history or frames accumulate on the GPU. The single retained temporal frame has fixed size. This avoids known graph-retention patterns by construction; it is not a universal guarantee against dependency or driver leaks. Long-run target-device monitoring remains necessary.

The benchmark synchronizes CUDA before timing and after each inference; reports median/p95 and peak allocated CUDA memory; excludes I/O and host/device transfers. CPU peak memory is deliberately null. A scalar operation-count comparison alone cannot rank real chip speed.
