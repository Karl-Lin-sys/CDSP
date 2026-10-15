"""Original wrappers using official OpenCV APIs; RGB visible images only unless stated."""
import numpy as np
import cv2

def rgb8(image):
    """O(P) validation, O(1) extra memory."""
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError("Expected nonempty uint8 HWC RGB")
    return image

def denoise(image, strength=5.0):
    """NLM conservative upper bound O(P*S^2*T^2), fixed S=21,T=7; O(P) workspace."""
    rgb8(image)
    if not np.isfinite(strength) or strength <= 0:
        raise ValueError("Positive finite strength required")
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    result = cv2.fastNlMeansDenoisingColored(bgr, None, strength, strength, 7, 21)
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

def resize(image, scale=2):
    """O(s^2 P) time/memory with fixed bicubic footprint; interpolation, not recovered detail."""
    rgb8(image)
    if type(scale) is not int or scale < 1:
        raise ValueError("Positive integer scale required")
    return cv2.resize(image, (image.shape[1]*scale, image.shape[0]*scale), interpolation=cv2.INTER_CUBIC)

def exposure_fusion(frames):
    """O(E*P) time/memory including geometric pyramids; display WDR, not radiometric HDR."""
    if len(frames) < 2:
        raise ValueError("At least two already aligned exposures required")
    for frame in frames:
        rgb8(frame)
        if frame.shape != frames[0].shape:
            raise ValueError("Exposure shapes differ")
    bgr = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in frames]
    fused = cv2.createMergeMertens().process(bgr)
    return np.clip(cv2.cvtColor(fused, cv2.COLOR_BGR2RGB), 0, 1)

class TemporalDenoiser:
    """Two-frame conservative temporal blend; no motion estimation, O(P) persistent state."""
    def __init__(self, alpha=0.25, threshold=0.03):
        """O(1). threshold is normalized intensity difference, fixed on validation data."""
        if not 0 <= alpha <= 1 or not 0 < threshold <= 1:
            raise ValueError("Invalid temporal parameters")
        self.alpha, self.threshold, self.previous = alpha, threshold, None

    def reset(self):
        """O(1); discard state at every new capture or scene cut."""
        self.previous = None

    def __call__(self, frame):
        """O(P) time/memory; reject changed pixels to reduce, not eliminate, ghosting."""
        rgb8(frame)
        current = frame.astype(np.float32)/255
        if self.previous is None:
            output = current
        else:
            if self.previous.shape != current.shape:
                raise ValueError("Call reset before resolution change")
            stable = np.max(np.abs(current-self.previous), axis=2, keepdims=True) < self.threshold
            output = np.where(stable, (1-self.alpha)*current+self.alpha*self.previous, current)
        self.previous = current.copy()
        return output

def thermal_correct(raw, gain, offset):
    """O(P) time/memory; apply supplied per-pixel calibration in its original units."""
    if raw.ndim != 2 or raw.shape != gain.shape or raw.shape != offset.shape or raw.size == 0:
        raise ValueError("Matching nonempty 2D calibration arrays required")
    if not all(np.isfinite(v).all() for v in (raw, gain, offset)) or (gain <= 0).any():
        raise ValueError("Finite calibration with positive gain required")
    return raw.astype(np.float32)*gain.astype(np.float32)+offset.astype(np.float32)
