"""Original artificial fixtures; no people, employer data, external images or weights."""
import argparse
import json
from pathlib import Path
import numpy as np

def generate(directory, scale=1, seed=42):
    """O(N*s^2*P) time/disk; O(s^2*P+N) memory. A plumbing test, not a camera benchmark."""
    if scale not in (1, 2, 4):
        raise ValueError("scale must be 1, 2 or 4")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for split in ("train", "val", "test"):
        for condition in ("low_noise", "high_noise"):
            for i in range(3):
                identity = f"{split}_{condition}_{i}"
                # Unique random scenes. No shared clean target across partitions.
                clean = rng.uniform(0.1, 0.9, (3, 16, 16)).astype(np.float32)
                sigma = 0.02 if condition == "low_noise" else 0.08
                noisy = np.clip(clean+rng.normal(0, sigma, clean.shape), 0, 1).astype(np.float32)
                target = np.repeat(np.repeat(clean, scale, axis=1), scale, axis=2)
                np.save(root/(identity+"_x.npy"), noisy)
                np.save(root/(identity+"_y.npy"), target)
                rows.append({"id": identity, "split": split, "capture": identity,
                             "scene": identity, "sensor": "synthetic", "stratum": condition,
                             "input": identity+"_x.npy", "target": identity+"_y.npy"})
    (root/"manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return root/"manifest.json"

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data")
    p.add_argument("--scale", type=int, default=1)
    a = p.parse_args()
    print(generate(a.output, a.scale))
