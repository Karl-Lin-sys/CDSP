"""Explicit full-frame, all-channel metrics; no hidden crop or colour transform."""
import math
import numpy as np

def errors(prediction, target):
    """O(P) time/memory. Return MSE, MAE and PSNR for fixed data range 1."""
    if prediction.shape != target.shape or prediction.size == 0:
        raise ValueError("Nonempty equal shapes required")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("Nonfinite values")
    if target.min() < 0 or target.max() > 1:
        raise ValueError("Target outside [0,1]")
    d = prediction.astype(np.float64) - target
    mse = float(np.mean(d*d))
    return {"mse": mse, "mae": float(np.mean(np.abs(d))),
            "psnr_db": None if mse == 0 else -10*math.log10(mse),
            "perfect": mse == 0}

def summarize(records, expected_strata):
    """O(N+G) time/memory; captures equal within strata, strata equal overall."""
    if not records or not expected_strata or len(set(expected_strata)) != len(expected_strata):
        raise ValueError("Records and unique expected strata required")
    groups = {}
    for r in records:
        if r["stratum"] not in expected_strata or not math.isfinite(r["mse"]) or r["mse"] < 0:
            raise ValueError("Unexpected stratum or invalid MSE")
        groups.setdefault((r["stratum"], r["capture"]), []).append(r["mse"])
    by_stratum = {}
    for (s, _), values in groups.items():
        by_stratum.setdefault(s, []).append(sum(values)/len(values))
    missing = sorted(set(expected_strata) - set(by_stratum))
    scores = {s: sum(v)/len(v) for s, v in by_stratum.items()}
    return {"stratum_capture_macro_mse": scores,
            "capture_counts": {s: len(v) for s, v in by_stratum.items()},
            "macro_mse": None if missing else sum(scores.values())/len(scores),
            "worst_stratum_mse": None if missing else max(scores.values()),
            "missing_strata": missing, "complete": not missing,
            "frame_micro_mse": sum(r["mse"] for r in records)/len(records)}

def capture_bootstrap(records, repeats=1000, seed=0):
    """O(R*N) time, O(N+R) memory. Cluster bootstrap; no frame independence claim."""
    if repeats < 2:
        raise ValueError("At least two bootstrap draws required")
    groups = {}
    for r in records:
        groups.setdefault(r["capture"], []).append(float(r["mse"]))
    if len(groups) < 2:
        return {"ci95": None, "reason": "Need at least two independent captures"}
    values = np.array([np.mean(v) for v in groups.values()])
    rng = np.random.default_rng(seed)
    draws = [np.mean(rng.choice(values, len(values), replace=True)) for _ in range(repeats)]
    return {"estimand": "equal-capture mean MSE (not stratum macro)",
            "ci95": np.quantile(draws, [0.025, 0.975]).tolist()}

def classification_counts(confusion):
    """O(K^2) time/memory. Optional downstream classifier; absent classes stay undefined."""
    a = np.asarray(confusion, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not a.size or not np.isfinite(a).all() or (a < 0).any():
        raise ValueError("Nonnegative finite square confusion matrix required")
    support = a.sum(axis=1)
    recalls = np.divide(a.diagonal(), support, out=np.full_like(support, np.nan), where=support>0)
    denom = support + a.sum(axis=0)
    f1 = np.divide(2*a.diagonal(), denom, out=np.full_like(support, np.nan), where=denom>0)
    complete = bool((support>0).all())
    return {"support": support.tolist(), "recall": [None if np.isnan(v) else float(v) for v in recalls],
            "balanced_accuracy": float(recalls.mean()) if complete else None,
            "macro_f1": float(f1.mean()) if complete else None, "complete": complete}
