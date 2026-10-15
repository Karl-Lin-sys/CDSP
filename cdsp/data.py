"""Manifest checks: split original captures BEFORE crops or augmentation."""
from pathlib import Path
import hashlib
import json
import numpy as np

def array_hash(path):
    """O(P) time and memory: hash decoded pixels, dtype and shape, not filenames."""
    a = np.load(path, allow_pickle=False)
    if a.dtype.hasobject or not np.issubdtype(a.dtype, np.number):
        raise ValueError("Numeric arrays required")
    h = hashlib.sha256(str((a.shape, a.dtype.str)).encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()

def read_manifest(path):
    """O(N + total pixels) time; O(N + largest image) memory."""
    path = Path(path)
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Manifest must be a nonempty list")
    seen_ids, ownership = set(), {}
    for r in rows:
        for key in ("id", "split", "capture", "scene", "sensor", "stratum", "input", "target"):
            if not isinstance(r.get(key), str) or not r[key].strip():
                raise ValueError("Missing nonempty field: " + key)
        if r["split"] not in {"train", "val", "test"} or r["id"] in seen_ids:
            raise ValueError("Invalid split or repeated id")
        seen_ids.add(r["id"])
        for key in ("input", "target"):
            p = (path.parent / r[key]).resolve()
            if not p.is_relative_to(path.parent.resolve()):
                raise ValueError("Data paths must stay inside manifest directory")
            r[key] = str(p)
        # Any shared source, scene, or decoded image must stay in one split.
        keys = [("capture", r["capture"]), ("scene", r["scene"])]
        keys += [("pixels", array_hash(r[k])) for k in ("input", "target")]
        for key in keys:
            previous = ownership.setdefault(key, r["split"])
            if previous != r["split"]:
                raise ValueError("Cross-split leakage: " + str(key))
    if {r["split"] for r in rows} != {"train", "val", "test"}:
        raise ValueError("All three splits required")
    return rows

def load_pair(row, scale):
    """O(P + s^2 P) time/memory; RGB or single-channel float32 CHW, [0,1]."""
    arrays = [np.load(row[k], allow_pickle=False) for k in ("input", "target")]
    for a in arrays:
        if a.dtype != np.float32 or a.ndim != 3 or a.shape[0] not in (1, 3):
            raise ValueError("Expected float32 CHW, one or three channels")
        if min(a.shape) < 1 or not np.isfinite(a).all() or a.min() < 0 or a.max() > 1:
            raise ValueError("Images must be finite and within [0,1]")
    x, y = arrays
    if y.shape != (x.shape[0], x.shape[1]*scale, x.shape[2]*scale):
        raise ValueError("Input/target geometry mismatch")
    return x, y
