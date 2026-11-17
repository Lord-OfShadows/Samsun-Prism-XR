"""Run predict() on every clip of a split and write outputs/predictions.jsonl.

  python scripts/make_predictions.py                # the test split of data/manifest.csv
  python scripts/make_predictions.py --workers 3    # faster; see the note on latency below
  python scripts/make_predictions.py --split val

One JSON object per line, in manifest order, exactly what predict() returns. EVERY clip id
in the split gets a line: if a clip cannot be read, its line has "prediction": null and an
"error" field, so the file is never silently short (and that clip scores as a miss).

predict() is called with the video path only, exactly as an evaluator would call it.

--workers N runs N copies of the pipeline side by side. The predictions are identical, but
the latency_ms values inside each line are then measured on a busy machine and are NOT
the benchmark. The latency we report comes from src/benchmark.py, which runs alone.
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402
from src.data import load_manifest  # noqa: E402


def _one(job):
    """(video_id, path) -> one output record. Never raises: the clip id must still appear."""
    video_id, path = job
    try:
        from predict import predict          # imported here so each worker builds its own pipeline
        rec = predict(path)
        rec["clip_id"] = video_id            # manifest id, keeps leading zeros
        return rec
    except Exception as e:  # noqa: BLE001
        return {"clip_id": video_id, "prediction": None, "confidence": 0.0, "top5": [],
                "error": f"{type(e).__name__}: {e}"[:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default=str(config.OUTPUT_DIR / "predictions.jsonl"))
    args = ap.parse_args()

    df = load_manifest()
    df = df[df.split == args.split].sort_values("video_id")
    if df.empty:
        sys.exit(f"no '{args.split}' clips in the manifest")
    jobs = list(zip(df.video_id, df.path))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    pool = Pool(args.workers) if args.workers > 1 else None
    results = pool.imap(_one, jobs, chunksize=1) if pool else map(_one, jobs)   # both keep order
    t0, n_err = time.perf_counter(), 0
    with open(args.out, "w", encoding="utf-8") as f:
        for i, rec in enumerate(results, 1):
            n_err += "error" in rec
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if i % 25 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} clips, {time.perf_counter() - t0:.0f}s", flush=True)
    if pool:
        pool.close()
    print(f"wrote {args.out}: {len(jobs)} lines, {n_err} errors, workers={args.workers}")


if __name__ == "__main__":   # required on Windows: worker processes re-import this file
    main()
