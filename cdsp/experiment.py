"""Train on train, select on validation, evaluate test only via a separate command."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from .data import read_manifest, load_pair
from .metrics import errors, summarize, capture_bootstrap
from .model import TinyRestorer

def seed_all(seed):
    """O(1) setup; determinism applies only within a compatible software/hardware stack."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False

def tensor_pair(row, scale, device):
    """O(P*s^2) I/O and memory; one image at a time, no unbounded tensor cache."""
    x, y = load_pair(row, scale)
    return (torch.from_numpy(x.copy()).unsqueeze(0).to(device),
            torch.from_numpy(y.copy()).unsqueeze(0).to(device))

@torch.inference_mode()
def evaluate(model, rows, device):
    """O(N*F) time, O(A+N) memory; F=forward cost, A=one-image activation memory."""
    model.eval()
    records = []
    for row in rows:
        x, y = tensor_pair(row, model.scale, device)
        pred = model(x).float().cpu().numpy()[0]
        records.append(dict(id=row["id"], capture=row["capture"], stratum=row["stratum"],
                            **errors(pred, y.cpu().numpy()[0])))
    return records

def train(manifest, output, scale=1, epochs=5, seed=7, device="cpu", patience=3):
    """O(E*N*F), memory O(A+W+N). Best checkpoint selected by validation macro MSE."""
    if epochs < 1 or patience < 1:
        raise ValueError("Positive epochs and patience required")
    seed_all(seed)
    rows = read_manifest(manifest)
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    strata = sorted({r["stratum"] for r in train_rows})
    if set(strata) != {r["stratum"] for r in val_rows}:
        raise ValueError("Validation must cover every training stratum")
    channels = load_pair(train_rows[0], scale)[0].shape[0]
    model = TinyRestorer(channels, 16, scale).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best, stale, best_state, history = float("inf"), 0, None, []
    # Inverse stratum AND capture frequency: equal total weight per stratum/capture.
    counts = {}
    captures = {}
    for r in train_rows:
        key = (r["stratum"], r["capture"])
        counts[key] = counts.get(key, 0) + 1
        captures.setdefault(r["stratum"], set()).add(r["capture"])
    for epoch in range(epochs):
        model.train()
        order = list(train_rows)
        random.shuffle(order)
        total = 0.0
        for row in order:
            x, y = tensor_pair(row, scale, device)
            optimizer.zero_grad(set_to_none=True)
            weight = len(train_rows)/(len(strata)*len(captures[row["stratum"]])*counts[(row["stratum"], row["capture"])])
            loss = torch.nn.functional.mse_loss(model(x), y) * weight
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            total += loss.detach().item()
        val = summarize(evaluate(model, val_rows, device), strata)["macro_mse"]
        history.append({"epoch": epoch+1, "train_weighted_mse": total/len(order), "validation_macro_mse": val})
        if val < best:
            best, stale = val, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= patience:
            break
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "channels": channels, "width": 16, "scale": scale,
                "strata": strata, "seed": seed}, output)
    report = {"history": history, "best_validation_macro_mse": best, "torch": torch.__version__,
              "numpy": np.__version__, "device": device, "seed": seed,
              "test_used_for_selection": False, "manifest_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest()}
    output.with_suffix(".train.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

def load_model(path, device="cpu"):
    """O(W) time/memory; only load checkpoints you own, weights-only loader."""
    c = torch.load(path, map_location="cpu", weights_only=True)
    m = TinyRestorer(c["channels"], c["width"], c["scale"])
    m.load_state_dict(c["state_dict"], strict=True)
    return m.to(device).eval(), c

def test(manifest, checkpoint, output, device="cpu"):
    """O(N*F+R*N) time, O(A+N+R) memory including clustered confidence interval."""
    model, config = load_model(checkpoint, device)
    rows = [r for r in read_manifest(manifest) if r["split"] == "test"]
    result = evaluate(model, rows, device)
    baseline = []
    for row in rows:
        x, y = tensor_pair(row, model.scale, "cpu")
        prediction = torch.nn.functional.interpolate(x, scale_factor=model.scale, mode="bilinear", align_corners=False)
        baseline.append(dict(id=row["id"], capture=row["capture"], stratum=row["stratum"],
                             **errors(prediction.numpy()[0], y.numpy()[0])))
    report = {"summary": summarize(result, config["strata"]),
              "baseline": {"name": "identity at scale 1; bilinear at scale 2/4",
                           "summary": summarize(baseline, config["strata"]), "records": baseline},
              "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
              "manifest_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
              "torch": torch.__version__, "device": device, "training_seed": config["seed"],
              "uncertainty": capture_bootstrap(result), "records": result,
              "protocol": "Full-frame float RGB or 1-channel; range 1; no clipping; no border shave"}
    Path(output).write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report

@torch.inference_mode()
def benchmark(model, shape=(1, 3, 64, 64), device="cpu", repeats=20):
    """O((5+R)*F) time, O(A+R) memory. Model-only latency, no disk/transfer."""
    if repeats < 1:
        raise ValueError("Positive repeat count required")
    model = model.to(device).eval()
    x = torch.zeros(shape, device=device)
    gpu = torch.device(device).type == "cuda"
    for _ in range(5):
        model(x)
    if gpu:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        model(x)
        if gpu:
            torch.cuda.synchronize(device)
        timings.append((time.perf_counter()-start)*1000)
    return {"median_ms": float(np.median(timings)), "p95_ms": float(np.quantile(timings, 0.95)),
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if gpu else None,
            "shape": list(shape), "device": str(device), "repeats": repeats,
            "torch": torch.__version__, "scope": "model only, warm, batch as supplied"}

def main():
    """CLI dispatch; complexity is that of selected operation."""
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["train", "test", "benchmark"])
    p.add_argument("--manifest", default="data/manifest.json")
    p.add_argument("--checkpoint", default="runs/model.pt")
    p.add_argument("--output", default="runs/evaluation.json")
    p.add_argument("--scale", type=int, default=1)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    if a.action == "train":
        result = train(a.manifest, a.checkpoint, a.scale, a.epochs, a.seed, a.device)
    elif a.action == "test":
        result = test(a.manifest, a.checkpoint, a.output, a.device)
    else:
        model, config = load_model(a.checkpoint, a.device)
        result = benchmark(model, (1, config["channels"], 64, 64), a.device)
    print(json.dumps(result, indent=2, allow_nan=False))

if __name__ == "__main__":
    main()
