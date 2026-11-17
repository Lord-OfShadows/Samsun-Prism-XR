"""The whole pipeline as ONE command: landmarks -> train -> INT8 -> every reported number.

  python scripts/run_all.py --wlasl-json data/wlasl/WLASL_v0.3.json --videos data/wlasl/videos

Retraining on a different word list is: edit configs/vocab.txt (one word per line, the
words must exist as glosses in the WLASL json), then run the command above. Nothing else
names the vocabulary; the number of classes is read from that file.

  --from train        start at a later step (earlier outputs must already exist)
  --only benchmark    run a single step

Steps, in order: models, extract, train, convert, predictions, benchmark, robustness, results
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEPS = ["models", "extract", "train", "convert", "predictions", "benchmark", "robustness", "results"]


def commands(args):
    py = sys.executable
    return {
        "models": [py, "scripts/get_mediapipe_models.py"],
        "extract": [py, "scripts/extract_all.py", "--wlasl-json", args.wlasl_json, "--videos", args.videos,
                    "--workers", str(args.workers)],
        "train": [py, "-m", "src.train", "--arch", args.arch],
        "convert": [py, "-m", "src.convert"],
        "predictions": [py, "scripts/make_predictions.py"],
        "benchmark": [py, "-m", "src.benchmark"],
        "robustness": [py, "-m", "src.robustness", "--workers", str(args.workers)],
        "results": [py, "scripts/make_results.py"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wlasl-json", help="path to WLASL_v0.3.json (needed for the extract step)")
    ap.add_argument("--videos", help="folder of <video_id>.mp4 (needed for the extract step)")
    ap.add_argument("--arch", default="conv_attn", choices=["conv_attn", "conv"])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--from", dest="start", choices=STEPS, default=STEPS[0])
    ap.add_argument("--only", choices=STEPS)
    args = ap.parse_args()

    todo = [args.only] if args.only else STEPS[STEPS.index(args.start):]
    if "extract" in todo and not (args.wlasl_json and args.videos):
        sys.exit("the extract step needs --wlasl-json and --videos (or start later with --from train)")
    cmds = commands(args)
    for step in todo:
        print(f"\n===== {step}: {' '.join(cmds[step][1:])} =====", flush=True)
        t0 = time.perf_counter()
        if subprocess.run(cmds[step], cwd=ROOT).returncode != 0:
            sys.exit(f"step '{step}' failed; fix it and rerun with --from {step}")
        print(f"===== {step} done in {time.perf_counter() - t0:.0f}s =====", flush=True)


if __name__ == "__main__":
    main()
