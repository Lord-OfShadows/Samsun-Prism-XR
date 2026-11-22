"""How much accuracy is lost when the video gets worse? Writes outputs/robustness.json.

  python -m src.robustness                # all test clips, 4 variants
  python -m src.robustness --limit 20     # quick check
  python -m src.robustness --split val --variants occluded
        # occluded VALIDATION set, for choosing between training setups. The test set is
        # then only used for the final number, as with clean accuracy.

For every test clip we damage the PIXELS, re-run MediaPipe on the damaged frames, and
classify with the INT8 model. Perturbing pixels (not landmarks) is the honest test: it
includes the landmark detector failing, which is what actually happens in bad conditions.

  dark      brightness x0.25                      (low light)
  rotated   15 degree tilt about the image centre (tilted head / camera)
  occluded  a grey box over 35% x 35% of the frame, at a per-clip random spot in the
            signing area, for the whole clip      (partial occlusion)
  mirrored  left-right flip                       (left-handed signer / selfie camera)

"Clean" accuracy is the same INT8 model on the cached, undamaged landmarks of the same
clips and the same frames, so the only difference is the perturbation.
drop_points = clean top-1 minus the mean top-1 of the four perturbed sets, in points.
Perturbed landmarks are cached in data/robustness/<variant>/ so reruns are fast.
"""
import argparse
import json
import sys
import zlib
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

VARIANTS = ("dark", "rotated", "occluded", "mirrored")
DARK_FACTOR, ROTATE_DEG, OCCLUDE_FRAC = 0.25, 15.0, 0.35


def perturb(frames, variant, seed=0):
    """Return damaged copies of a clip's frames (list of BGR uint8 arrays)."""
    h, w = frames[0].shape[:2]
    if variant == "dark":
        return [(f.astype(np.float32) * DARK_FACTOR).astype(np.uint8) for f in frames]
    if variant == "rotated":
        m = cv2.getRotationMatrix2D((w / 2, h / 2), ROTATE_DEG, 1.0)
        return [cv2.warpAffine(f, m, (w, h), borderMode=cv2.BORDER_CONSTANT) for f in frames]
    if variant == "occluded":
        rng = np.random.default_rng(seed)
        bw, bh = int(OCCLUDE_FRAC * w), int(OCCLUDE_FRAC * h)
        cx, cy = rng.uniform(0.3, 0.7) * w, rng.uniform(0.35, 0.75) * h   # chest / hands region
        x0, y0 = int(np.clip(cx - bw / 2, 0, w - bw)), int(np.clip(cy - bh / 2, 0, h - bh))
        out = [f.copy() for f in frames]
        for f in out:
            f[y0:y0 + bh, x0:x0 + bw] = 127
        return out
    if variant == "mirrored":
        return [cv2.flip(f, 1) for f in frames]
    raise ValueError(variant)


def read_exact_frames(path, frame_idx):
    """Decode exactly the frames listed in a cached clip's frame_idx (same frames as clean)."""
    want, frames, i = set(int(k) for k in frame_idx), [], 0
    cap = cv2.VideoCapture(str(path))
    last = max(want)
    while i <= last:
        ok, frame = cap.read()
        if not ok:
            break
        if i in want:
            frames.append(frame)
        i += 1
    cap.release()
    return frames


_extractor = None


def _init_worker():
    global _extractor
    from src.landmarks import LandmarkExtractor
    _extractor = LandmarkExtractor(config.HAND_MODEL, config.POSE_MODEL)


def _process(job):
    """Extract landmarks for one (clip, variant). Returns (video_id, variant, error or None)."""
    from src.landmarks import shrink
    out = Path(job["out"])
    if out.exists():
        return job["video_id"], job["variant"], None
    try:
        with np.load(job["clean"]) as d:
            frame_idx, fps = d["frame_idx"], float(d["fps"])
        frames = shrink(read_exact_frames(job["path"], frame_idx))
        if not frames:
            raise IOError("no frames decoded")
        seed = zlib.crc32(job["video_id"].encode())        # stable per clip, across runs
        frames = perturb(frames, job["variant"], seed)
        hands, pose = _extractor.extract_frames(frames, fps)
        np.savez_compressed(out, hands=hands, pose=pose, width=np.int32(frames[0].shape[1]),
                            height=np.int32(frames[0].shape[0]))
        return job["video_id"], job["variant"], None
    except Exception as e:  # noqa: BLE001 - a broken clip is scored as a miss, not a crash
        return job["video_id"], job["variant"], f"{type(e).__name__}: {e}"[:200]


def accuracy(clf, raws, labels):
    """(top1, top5). raws[i] is a raw clip dict or None (extraction failed = miss)."""
    from src.normalize import normalize
    hit1 = hit5 = 0
    for raw, y in zip(raws, labels):
        if raw is None:
            continue
        order = np.argsort(-clf.probs(normalize(raw["hands"], raw["pose"], raw["width"])))
        hit1 += order[0] == y
        hit5 += y in order[:5]
    return hit1 / len(labels), hit5 / len(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="only the first N test clips (quick check)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    args = ap.parse_args()
    # test keeps its original cache folder; other splits get their own sub-folder
    base = config.ROBUSTNESS_DIR if args.split == "test" else config.ROBUSTNESS_DIR / args.split

    from src.classifier import Int8Classifier
    from src.data import load_manifest, load_raw

    df = load_manifest()
    df = df[(df.split == args.split) & df.status.isin(["ok", "no_hands"])].sort_values("video_id")
    if args.limit:
        df = df.head(args.limit)
    labels = df.label.to_numpy()

    jobs = []
    for v in args.variants:
        (base / v).mkdir(parents=True, exist_ok=True)
        jobs += [dict(video_id=r.video_id, variant=v, path=r.path,
                      clean=str(config.LANDMARK_DIR / f"{r.video_id}.npz"),
                      out=str(base / v / f"{r.video_id}.npz")) for r in df.itertuples()]
    errors = []
    with Pool(args.workers, initializer=_init_worker) as pool:
        for i, (vid, v, err) in enumerate(pool.imap_unordered(_process, jobs, chunksize=1), 1):
            if err:
                errors.append({"video_id": vid, "variant": v, "error": err})
            if i % 50 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} perturbed clips extracted", flush=True)

    clf = Int8Classifier(config.INT8_MODEL)
    clean1, clean5 = accuracy(clf, [load_raw(v) for v in df.video_id], labels)
    result = {"clips": int(len(df)), "clean_top1": clean1, "clean_top5": clean5, "variants": {},
              "settings": {"dark_factor": DARK_FACTOR, "rotate_deg": ROTATE_DEG,
                           "occlude_frac": OCCLUDE_FRAC}, "extraction_errors": errors}
    print(f"\n{'set':10s} {'top1':>7s} {'top5':>7s} {'drop':>7s}")
    print(f"{'clean':10s} {100 * clean1:7.2f} {100 * clean5:7.2f}")
    for v in args.variants:
        raws = []
        for vid in df.video_id:
            p = base / v / f"{vid}.npz"
            raws.append(load_raw(vid, base / v) if p.exists() else None)
        t1, t5 = accuracy(clf, raws, labels)
        result["variants"][v] = {"top1": t1, "top5": t5, "drop_points": round(100 * (clean1 - t1), 3)}
        print(f"{v:10s} {100 * t1:7.2f} {100 * t5:7.2f} {100 * (clean1 - t1):7.2f}")
    mean1 = float(np.mean([r["top1"] for r in result["variants"].values()]))
    result["perturbed_mean_top1"] = mean1
    result["drop_points"] = round(100 * (clean1 - mean1), 3)
    print(f"{'mean':10s} {100 * mean1:7.2f} {'':7s} {result['drop_points']:7.2f}")
    if errors:
        print(f"{len(errors)} perturbed clips failed extraction and were scored as misses")

    full = not args.limit and list(args.variants) == list(VARIANTS)
    if full:                # a partial run must not overwrite the real result
        name = "robustness.json" if args.split == "test" else f"robustness_{args.split}.json"
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (config.OUTPUT_DIR / name).write_text(json.dumps(result, indent=1))
        print(f"wrote {config.OUTPUT_DIR / name}")


if __name__ == "__main__":
    main()
