"""Evaluates every model on the SAME held-out test set (OffensEval-2019 test, 860 tweets).

Metric : macro-F1 (classes are imbalanced: 620 clean / 240 offensive), plus accuracy.
Size   : bytes on disk of the files the model needs at inference time.
Latency: end-to-end time for ONE message (normalization + model), CPU only, 1 thread,
         averaged over all 860 test messages (after 20 warm-up calls).

Run:  python -m src.evaluate
"""
import json
import platform
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score

from .config import CPU_THREADS, LATENCY_BUDGET_MS, RESULTS_DIR
from .data import load_splits
from .predictors import MODELS, load_model, model_size_mb


def cpu_name():
    try:
        import subprocess
        out = subprocess.run(["wmic", "cpu", "get", "name"], capture_output=True, text=True).stdout.split("\n")
        return [l.strip() for l in out if l.strip() and l.strip() != "Name"][0]
    except Exception:
        return platform.processor()


def main():
    torch.set_num_threads(CPU_THREADS)
    test_x, test_y = load_splits()["test"]
    rows, examples = [], {}
    for name in MODELS:
        print(f"\n=== {name}")
        model = load_model(name)
        for t in test_x[:20]:
            model.predict_proba(t)  # warm-up
        probs, times = [], []
        for t in test_x:
            t0 = time.perf_counter()
            probs.append(model.predict_proba(t))
            times.append((time.perf_counter() - t0) * 1000)
        pred = (np.array(probs) >= 0.5).astype(int)
        times = np.array(times)
        row = {
            "model": name,
            "precision": MODELS[name][2],
            "macro_f1": round(f1_score(test_y, pred, average="macro"), 4),
            "accuracy": round(accuracy_score(test_y, pred), 4),
            "offensive_f1": round(f1_score(test_y, pred, pos_label=1), 4),
            "size_mb": round(model_size_mb(name), 3),
            "latency_ms_mean": round(float(times.mean()), 2),
            "latency_ms_p95": round(float(np.percentile(times, 95)), 2),
            "within_budget": bool(times.mean() <= LATENCY_BUDGET_MS),
            "runs": len(times),
        }
        rows.append(row)
        examples[name] = pred.tolist()
        print(classification_report(test_y, pred, target_names=["clean", "offensive"], digits=4))
        print(json.dumps(row, indent=1))

    hw = f"{cpu_name()}, PyTorch {torch.__version__}, {CPU_THREADS} CPU thread(s), batch size 1"
    (RESULTS_DIR / "results.json").write_text(json.dumps({"hardware": hw, "budget_ms": LATENCY_BUDGET_MS, "rows": rows}, indent=1))
    (RESULTS_DIR / "test_predictions.json").write_text(json.dumps({"labels": test_y, "texts": test_x, "preds": examples}))

    md = [
        f"Test set: OffensEval-2019 test split, {len(test_y)} tweets ({sum(test_y)} offensive). ",
        f"Latency: mean over {len(test_x)} single-message runs on {hw}. Budget: {LATENCY_BUDGET_MS:.0f} ms/message.\n",
        "| Model | Precision | Macro-F1 | Accuracy | Size (MB) | Latency mean (ms) | Latency p95 (ms) | Within budget |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        md.append(
            f"| {r['model']} | {r['precision']} | {r['macro_f1']:.4f} | {r['accuracy']:.4f} | {r['size_mb']:.3f} | "
            f"{r['latency_ms_mean']:.2f} | {r['latency_ms_p95']:.2f} | {'yes' if r['within_budget'] else 'no'} |"
        )
    (RESULTS_DIR / "results.md").write_text("\n".join(md) + "\n")
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
