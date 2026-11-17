"""Rewrite the Results block of README.md from the files in outputs/.

  python scripts/update_readme.py

Everything between <!-- RESULTS:START --> and <!-- RESULTS:END --> is replaced. Each
table is built from a json or csv that another script wrote, so no number in the README
is typed by hand. A table whose input file does not exist yet is replaced by a line
saying which script to run.
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def _json(name):
    p = config.OUTPUT_DIR / name
    return json.loads(p.read_text()) if p.exists() else None


def pct(x):
    return f"{100 * x:.2f} %"


def results_table():
    p = config.OUTPUT_DIR / "results.csv"
    if not p.exists():
        return ["Not generated yet: run `python scripts/make_results.py`."]
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    out = ["### Organisers' results table (`outputs/results.csv`)", "",
           "| Metric | Target | Measured | Pass |", "|---|---|---|---|"]
    for r in rows:
        value = r["value"] if r["value"] == "not measured" else f"{r['value']} {r['unit']}"
        out.append(f"| `{r['metric']}` | {r['target']} | {value} | {r['pass']} |")
    out += ["", "Notes per row:", ""]
    out += [f"- `{r['metric']}`: {r['notes']}" for r in rows]
    return out


def accuracy_section(q, t):
    if not q:
        return ["### Accuracy", "", "Not generated yet: run `python -m src.train` then `python -m src.convert`."]
    out = ["### Accuracy: float model against INT8 (`outputs/quantization.json`)", "",
           "Measured on cached landmarks of the official WLASL100 splits. Every clip in the "
           "split counts, including clips where no hand was found.", "",
           "| Split | Clips | Float top-1 | INT8 top-1 | Float top-5 | INT8 top-5 | Same top word |",
           "|---|---|---|---|---|---|---|"]
    for s in ("val", "test"):
        r = q[s]
        out.append(f"| {s} | {r['clips']} | {pct(r['float_top1'])} | {pct(r['int8_top1'])} | "
                   f"{pct(r['float_top5'])} | {pct(r['int8_top5'])} | {pct(r['same_top1_prediction'])} |")
    out += ["", f"- Parameters: {q['parameters']:,}",
            f"- Float model file: {q['float_model_bytes'] / 1e6:.2f} MB; "
            f"INT8 model file: {q['int8_model_bytes'] / 1e6:.2f} MB",
            f"- Float tensors left in the INT8 model: {q['float_tensors_left']} (0 means fully integer)",
            f"- Representative set: {q['representative_clips']} training clips"]
    if t:
        out.append(f"- Training: {t['train_clips']} clips, best validation epoch {t['best_epoch']} of "
                   f"{t['epochs_run']} run, {t['train_seconds'] / 60:.0f} minutes on CPU")
    return out


def benchmark_section(b):
    if not b:
        return ["### Latency and size", "", "Not measured yet: run `python -m src.benchmark`."]
    e, s = b["end_to_end_ms"], b["size"]
    rows = [("One classifier call (isolated)", b["classifier_only_ms"]),
            ("Stage: decode video", e["decode"]), ("Stage: landmarks, whole clip", e["landmarks"]),
            ("Stage: normalise", e["normalize"]), ("Stage: classifier", e["classifier"]),
            ("Landmarks per frame", e["per_frame_landmarks"]), ("**End to end, one clip**", e["total"])]
    out = ["### Latency (`outputs/benchmark.json`)", "",
           f"Device: {b['device']['cpu']}, {b['device']['os']}. Runtime: {b['device']['runtime']}, "
           f"{b['classifier_threads']} thread for the classifier. "
           f"{b['classifier_only_ms']['runs']} timed runs after {b['warmup_runs']} warm-up runs, "
           f"over {b['clips_timed']} test clips (median {b['frames_per_clip']['p50']:.0f} frames per clip).", "",
           "| Measurement | p50 (ms) | p95 (ms) | max (ms) |", "|---|---|---|---|"]
    out += [f"| {n} | {v['p50']:.2f} | {v['p95']:.2f} | {v['max']:.2f} |" for n, v in rows]
    out += ["", "The brief does not say whether its 50 ms ceiling means one classifier call, one "
                "frame, or a whole clip, so all three are reported. This is a laptop CPU, not a phone.", "",
            "### Model size", "", "| File | Size (MB) |", "|---|---|"]
    out += [f"| {k} | {v / 1e6:.2f} |" for k, v in s["files_bytes"].items()]
    out += [f"| **Total** | **{s['total_mb']:.2f}** |", "",
            f"Budget {s['budget_mb']:.0f} MB: {'within budget' if s['within_budget'] else 'OVER budget'}. "
            f"MB = 1,000,000 bytes."]
    if "note" in s:
        out.append(s["note"])
    return out


def robustness_section(r):
    if not r:
        return ["### Robustness", "", "Not measured yet: run `python -m src.robustness`."]
    st = r["settings"]
    out = ["### Robustness (`outputs/robustness.json`)", "",
           f"The pixels of all {r['clips']} test clips are damaged, MediaPipe is re-run, and the INT8 "
           f"model classifies the result.", "",
           "| Test set | Top-1 | Top-5 | Drop in top-1 (points) |", "|---|---|---|---|",
           f"| Clean | {pct(r['clean_top1'])} | {pct(r['clean_top5'])} | |"]
    names = {"dark": f"Dark (brightness x{st['dark_factor']})",
             "rotated": f"Rotated ({st['rotate_deg']:.0f} degrees)",
             "occluded": f"Occluded (box over {st['occlude_frac']:.0%} x {st['occlude_frac']:.0%} of the frame)",
             "mirrored": "Mirrored (left-right flip)"}
    for k, v in r["variants"].items():
        out.append(f"| {names.get(k, k)} | {pct(v['top1'])} | {pct(v['top5'])} | {v['drop_points']:.2f} |")
    out.append(f"| **Mean of the four** | **{pct(r['perturbed_mean_top1'])}** | | **{r['drop_points']:.2f}** |")
    return out


def main():
    blocks = [results_table(), accuracy_section(_json("quantization.json"), _json("train_log.json")),
              benchmark_section(_json("benchmark.json")), robustness_section(_json("robustness.json")),
              ["### Not measured", "",
               "`avg_power` and `thermal_drift_10min` need a run on a phone. There is no Android "
               "build (it is a stretch goal), so they are reported as not measured, never estimated."]]
    body = "\n\n".join("\n".join(b) for b in blocks)
    readme = config.ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    if START not in text or END not in text:
        sys.exit("README.md has lost its RESULTS markers")
    head, rest = text.split(START, 1)
    tail = rest.split(END, 1)[1]
    readme.write_text(f"{head}{START}\n{body}\n{END}{tail}", encoding="utf-8", newline="\n")
    print("README.md results block updated")


if __name__ == "__main__":
    main()
