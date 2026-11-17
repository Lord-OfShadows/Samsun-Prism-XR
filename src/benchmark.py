"""Measure latency and model size. Writes outputs/benchmark.json.

  python -m src.benchmark                 # 200 timed runs per measurement
  python -m src.benchmark --runs 20       # quick check

Latency (p95 over N timed runs, after warm-up runs that are thrown away):
  classifier   one INT8 classifier call on an already-normalised clip
  per stage    decode / landmarks / normalise / classifier inside predict()
  per frame    landmark time divided by the frames processed (the cost of one new frame)
  end to end   the whole predict() call for one clip
The brief does not say which of these the 50 ms ceiling means, so all are reported.

Size: every model file the pipeline loads (our classifier + both MediaPipe models),
summed, against the 20 MB budget.

Numbers depend on the machine; the device, OS and library versions are saved alongside.
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402


def stats(ms):
    a = np.asarray(ms, np.float64)
    return {"p50": round(float(np.percentile(a, 50)), 3), "p95": round(float(np.percentile(a, 95)), 3),
            "mean": round(float(a.mean()), 3), "min": round(float(a.min()), 3),
            "max": round(float(a.max()), 3), "runs": int(len(a))}


def model_sizes():
    files = {"classifier_int8": config.INT8_MODEL, "mediapipe_hand_landmarker": config.HAND_MODEL,
             "mediapipe_pose_landmarker_lite": config.POSE_MODEL}
    sizes = {k: p.stat().st_size for k, p in files.items()}
    total_mb = sum(sizes.values()) / 1e6
    hands_only_mb = (sizes["classifier_int8"] + sizes["mediapipe_hand_landmarker"]) / 1e6
    out = {"files_bytes": sizes, "total_mb": round(total_mb, 3), "budget_mb": config.SIZE_BUDGET_MB,
           "within_budget": bool(total_mb <= config.SIZE_BUDGET_MB),
           "classifier_only_mb": round(sizes["classifier_int8"] / 1e6, 3),
           "unit": "MB = 1,000,000 bytes"}
    if not out["within_budget"]:
        out["note"] = (f"Over budget. Fallback from the PRD: drop the pose model and keep hands only "
                       f"({hands_only_mb:.2f} MB). NOT applied automatically; it needs retraining "
                       f"on hands-only features and re-measuring accuracy.")
    return out


def bench_classifier(runs, warmup):
    from src.classifier import Int8Classifier
    from src.normalize import N_FEATURES
    clf = Int8Classifier(config.INT8_MODEL)
    x = np.random.default_rng(0).uniform(-1, 1, (config.SEQ_LEN, N_FEATURES)).astype(np.float32)
    ms = []
    for i in range(warmup + runs):
        t = time.perf_counter()
        clf.probs(x)                       # includes float->int8 input scaling
        if i >= warmup:
            ms.append((time.perf_counter() - t) * 1e3)
    return stats(ms), clf.num_threads


def bench_end_to_end(videos, runs, warmup):
    from predict import run_stages
    per = {k: [] for k in ("decode", "landmarks", "normalize", "classifier", "total", "per_frame_landmarks")}
    frames = []
    for i in range(warmup + runs):
        _, ms, n = run_stages(videos[i % len(videos)])
        if i >= warmup:
            for k in ms:
                per[k].append(ms[k])
            per["per_frame_landmarks"].append(ms["landmarks"] / n)
            frames.append(n)
        if (i + 1) % 20 == 0:
            print(f"  end-to-end run {i + 1}/{warmup + runs}", flush=True)
    return {k: stats(v) for k, v in per.items()}, stats(frames)


def device_info():
    import ai_edge_litert
    import mediapipe
    cpu = platform.processor()
    try:   # friendlier CPU name on Windows
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
            cpu = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
    except Exception:  # noqa: BLE001 - not Windows, keep platform.processor()
        pass
    return {"cpu": cpu, "os": platform.platform(), "python": platform.python_version(),
            "runtime": f"LiteRT (ai-edge-litert {ai_edge_litert.__version__}), CPU, XNNPACK delegate",
            "mediapipe": mediapipe.__version__}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--videos", nargs="*", help="clips to time; default: first 20 test clips in the manifest")
    args = ap.parse_args()

    videos = args.videos
    if not videos:
        from src.data import load_manifest
        df = load_manifest()
        videos = df[df.split == "test"].sort_values("video_id").path.head(20).tolist()
    videos = [v for v in videos if Path(v).exists()]
    if not videos:
        sys.exit("no videos to time; pass --videos or run scripts/extract_all.py first")

    result = {"device": device_info(), "size": model_sizes()}
    print("timing classifier ...", flush=True)
    result["classifier_only_ms"], threads = bench_classifier(args.runs, args.warmup)
    result["classifier_threads"] = threads
    print(f"timing end to end on {len(videos)} clips ...", flush=True)
    result["end_to_end_ms"], result["frames_per_clip"] = bench_end_to_end(videos, args.runs, args.warmup)
    result["clips_timed"] = len(videos)
    result["warmup_runs"] = args.warmup
    result["latency_budget_ms"] = config.LATENCY_BUDGET_MS

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (config.OUTPUT_DIR / "benchmark.json").write_text(json.dumps(result, indent=1))

    s = result["size"]
    print(f"\ndevice: {result['device']['cpu']} | {result['device']['runtime']}")
    print(f"\nmodel size (budget {s['budget_mb']} MB)")
    for k, b in s["files_bytes"].items():
        print(f"  {k:32s} {b / 1e6:7.3f} MB")
    print(f"  {'TOTAL':32s} {s['total_mb']:7.3f} MB  -> {'within' if s['within_budget'] else 'OVER'} budget")
    if "note" in s:
        print("  " + s["note"])
    print(f"\nlatency in ms over {args.runs} runs (budget {config.LATENCY_BUDGET_MS} ms)")
    print(f"  {'measurement':28s} {'p50':>9s} {'p95':>9s} {'max':>9s}")
    rows = [("classifier call (isolated)", result["classifier_only_ms"])]
    names = {"decode": "stage: decode video", "landmarks": "stage: landmarks (clip)",
             "normalize": "stage: normalise", "classifier": "stage: classifier",
             "per_frame_landmarks": "landmarks per frame", "total": "END TO END (one clip)"}
    rows += [(names[k], result["end_to_end_ms"][k]) for k in names]
    for name, st in rows:
        print(f"  {name:28s} {st['p50']:9.2f} {st['p95']:9.2f} {st['max']:9.2f}")
    print(f"\nwrote {config.OUTPUT_DIR / 'benchmark.json'}")


if __name__ == "__main__":
    main()
