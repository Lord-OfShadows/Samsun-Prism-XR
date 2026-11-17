"""Build outputs/results.csv, the organisers' six-row table, from files other scripts wrote.

  python scripts/make_results.py

Inputs (each produced by a script in this repo, never typed by hand):
  outputs/predictions.jsonl  + data/manifest.csv   -> top1_accuracy (and top-5 in notes)
  outputs/benchmark.json                           -> latency_p95, model_size
  outputs/robustness.json                          -> accuracy_drop_robustness
avg_power and thermal_drift_10min need an on-phone measurement we do not have. They are
written as "not measured", never estimated. Any row whose input file is missing is also
"not measured", with the script to run in the notes.

Columns: metric, target, value, unit, pass, notes.   pass is PASS / FAIL / not measured.
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

NOT_MEASURED = "not measured"
FIELDS = ["metric", "target", "value", "unit", "pass", "notes"]


def _json(name):
    p = config.OUTPUT_DIR / name
    return json.loads(p.read_text()) if p.exists() else None


def score_predictions(lines, labels, vocab):
    """lines: parsed predictions.jsonl. labels: {clip_id: class index} for the split.
    Returns (top1, top5, n). A clip with no usable prediction, or missing from the file,
    counts as wrong: the denominator is always every clip in the split."""
    by_id = {r["clip_id"]: r for r in lines}
    hit1 = hit5 = 0
    for clip_id, label in labels.items():
        r, word = by_id.get(clip_id), vocab[label]
        if not r or not r.get("prediction"):
            continue
        hit1 += r["prediction"] == word
        hit5 += word in [t["word"] for t in r.get("top5", [])]
    n = len(labels)
    return hit1 / n, hit5 / n, n


def verdict(value, limit, higher_is_better):
    ok = value >= limit if higher_is_better else value <= limit
    return "PASS" if ok else "FAIL"


def accuracy_row():
    pred_path = config.OUTPUT_DIR / "predictions.jsonl"
    if not (pred_path.exists() and config.MANIFEST_PATH.exists()):
        return ["top1_accuracy", ">= 70 %", NOT_MEASURED, "%", NOT_MEASURED,
                "run scripts/make_predictions.py (needs the dataset)"]
    from src.data import load_manifest
    lines = [json.loads(s) for s in pred_path.read_text(encoding="utf-8").splitlines() if s.strip()]
    df = load_manifest()
    ids = {r["clip_id"] for r in lines}
    split = df[df.video_id.isin(ids)].split.mode()[0]       # the split the file was made from
    df = df[df.split == split]
    top1, top5, n = score_predictions(lines, dict(zip(df.video_id, df.label)), config.load_vocab())
    return ["top1_accuracy", ">= 70 %", f"{100 * top1:.2f}", "%", verdict(100 * top1, 70, True),
            f"INT8 model via predict() on all {n} clips of the WLASL100 {split} split; "
            f"top-5 = {100 * top5:.2f} %"]


def latency_row(bench):
    if not bench:
        return ["latency_p95", "<= 50 ms", NOT_MEASURED, "ms", NOT_MEASURED, "run python -m src.benchmark"]
    e2e, clf = bench["end_to_end_ms"], bench["classifier_only_ms"]
    options = {"end_to_end": ("whole clip incl. landmarks", e2e["total"]["p95"]),
               "per_frame": ("landmarks for one frame", e2e["per_frame_landmarks"]["p95"]),
               "classifier": ("one classifier call", clf["p95"])}
    label, value = options[config.LATENCY_METRIC_FOR_RESULTS]
    notes = (f"value = {label} (config.LATENCY_METRIC_FOR_RESULTS). All p95 over {clf['runs']} runs: "
             f"classifier call {clf['p95']:.2f} ms; landmarks per frame {e2e['per_frame_landmarks']['p95']:.2f} ms; "
             f"whole clip {e2e['total']['p95']:.2f} ms ({bench['frames_per_clip']['p50']:.0f} frames). "
             f"Device: {bench['device']['cpu']}, {bench['device']['runtime']}")
    return ["latency_p95", "<= 50 ms", f"{value:.2f}", "ms",
            verdict(value, config.LATENCY_BUDGET_MS, False), notes]


def size_row(bench):
    if not bench:
        return ["model_size", "<= 25 MB", NOT_MEASURED, "MB", NOT_MEASURED, "run python -m src.benchmark"]
    s = bench["size"]
    parts = "; ".join(f"{k} {v / 1e6:.2f}" for k, v in s["files_bytes"].items())
    return ["model_size", "<= 25 MB", f"{s['total_mb']:.2f}", "MB",
            verdict(s["total_mb"], config.SIZE_BUDGET_MB, False),
            f"sum of every model file loaded ({parts}); pass is judged against our stricter "
            f"{config.SIZE_BUDGET_MB:.0f} MB design budget"]


def robustness_row(rob):
    if not rob:
        return ["accuracy_drop_robustness", "<= 5 points", NOT_MEASURED, "points", NOT_MEASURED,
                "run python -m src.robustness (needs the dataset)"]
    per = "; ".join(f"{k} {100 * v['top1']:.2f} %" for k, v in rob["variants"].items())
    drop = rob["drop_points"]
    return ["accuracy_drop_robustness", "<= 5 points", f"{drop:.2f}", "points", verdict(drop, 5, False),
            f"clean {100 * rob['clean_top1']:.2f} % minus mean of perturbed sets "
            f"{100 * rob['perturbed_mean_top1']:.2f} % ({per}); {rob['clips']} test clips each"]


def main():
    bench, rob = _json("benchmark.json"), _json("robustness.json")
    phone = "needs an on-phone run (Android stretch goal not built); not estimated"
    rows = [accuracy_row(), latency_row(bench), size_row(bench), robustness_row(rob),
            ["avg_power", "<= 500 mW", NOT_MEASURED, "mW", NOT_MEASURED, phone],
            ["thermal_drift_10min", "<= 20 %", NOT_MEASURED, "%", NOT_MEASURED, phone]]
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = config.OUTPUT_DIR / "results.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        w.writerows(rows)
    print(f"wrote {out}\n")
    for r in rows:
        print(f"{r[0]:26s} {r[2]:>13s} {r[3]:7s} target {r[1]:12s} {r[4]}")


if __name__ == "__main__":
    main()
