"""Run predict() on every clip of a split and write outputs/predictions.jsonl.

  python scripts/make_predictions.py                # the test split of data/manifest.csv
  python scripts/make_predictions.py --split val

One JSON object per line, in manifest order, exactly what predict() returns. EVERY clip id
in the split gets a line: if a clip cannot be read, its line has "prediction": null and an
"error" field, so the file is never silently short (and that clip scores as a miss).

predict() is called with the video path only, exactly as an evaluator would call it.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from predict import predict  # noqa: E402
from src import config  # noqa: E402
from src.data import load_manifest  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=str(config.OUTPUT_DIR / "predictions.jsonl"))
    args = ap.parse_args()

    df = load_manifest()
    df = df[df.split == args.split].sort_values("video_id")
    if df.empty:
        sys.exit(f"no '{args.split}' clips in the manifest")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    t0, n_err = time.perf_counter(), 0
    with open(args.out, "w", encoding="utf-8") as f:
        for i, row in enumerate(df.itertuples(), 1):
            try:
                rec = predict(row.path)
                rec["clip_id"] = row.video_id          # manifest id, keeps leading zeros
            except Exception as e:  # noqa: BLE001 - the clip id must still appear
                n_err += 1
                rec = {"clip_id": row.video_id, "prediction": None, "confidence": 0.0,
                       "top5": [], "error": f"{type(e).__name__}: {e}"[:200]}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if i % 25 == 0 or i == len(df):
                print(f"  {i}/{len(df)} clips, {time.perf_counter() - t0:.0f}s", flush=True)
    print(f"wrote {args.out}: {len(df)} lines, {n_err} errors")


if __name__ == "__main__":
    main()
