"""Cache MediaPipe landmarks for every WLASL clip in the vocabulary.

In:  WLASL_v0.3.json, a folder of <video_id>.mp4 files, configs/vocab.txt
Out: data/landmarks/<video_id>.npz  (arrays described in src/landmarks.py)
     data/manifest.csv              one row per clip: split, label, status, detection rates

Resumable: clips that already have an .npz are skipped unless --overwrite.

Examples
  python scripts/extract_all.py --wlasl-json D/WLASL_v0.3.json --videos D/videos --write-vocab 100
  python scripts/extract_all.py --wlasl-json D/WLASL_v0.3.json --videos D/videos --limit 5 --preview
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `src` imports from scripts/
from src import config

_extractor = None  # one per worker process; MediaPipe objects can't be pickled across processes


def _init_worker():
    global _extractor
    from src.landmarks import LandmarkExtractor
    _extractor = LandmarkExtractor(config.HAND_MODEL, config.POSE_MODEL)


def _process(job):
    """Run one clip. Always returns a manifest row, never raises (one bad file must not kill the run)."""
    row = {k: job[k] for k in ("video_id", "gloss", "label", "split", "signer_id", "path")}
    out = Path(job["out"])
    t0 = time.perf_counter()
    try:
        if out.exists() and not job["overwrite"]:
            d = dict(np.load(out))
            row["status_note"] = "cached"
        else:
            r = _extractor.extract(job["path"], job["frame_start"], job["frame_end"])
            preview = r.pop("_preview_frame")
            d = r
            np.savez_compressed(out, **d)
            if job["preview_dir"]:
                import cv2
                from src.landmarks import draw_preview
                t = len(d["hands"]) // 2
                cv2.imwrite(str(Path(job["preview_dir"]) / f"{job['video_id']}.png"),
                            draw_preview(preview, d["hands"][t], d["pose"][t]))
            row["status_note"] = str(d["bounds_mode"])
    except Exception as e:  # noqa: BLE001 - we want the message in the manifest
        row.update(status="read_error", error=f"{type(e).__name__}: {e}"[:200])
        return row

    hands, pose = d["hands"], d["pose"]
    has_l = np.isfinite(hands[:, 0, 0, 0])
    has_r = np.isfinite(hands[:, 1, 0, 0])
    row.update(
        status="ok" if (has_l | has_r).any() else "no_hands",
        n_frames=len(hands),
        frac_left=round(float(has_l.mean()), 3),
        frac_right=round(float(has_r.mean()), 3),
        frac_any_hand=round(float((has_l | has_r).mean()), 3),
        frac_pose=round(float(np.isfinite(pose[:, 0, 0]).mean()), 3),
        seconds=round(time.perf_counter() - t0, 3),
    )
    return row


def build_jobs(meta, vocab, args):
    gloss_to_label = {g: i for i, g in enumerate(vocab)}
    jobs = []
    for entry in meta:
        if entry["gloss"] not in gloss_to_label:
            continue
        for inst in entry["instances"]:
            vid = inst["video_id"]
            jobs.append(dict(
                video_id=vid, gloss=entry["gloss"], label=gloss_to_label[entry["gloss"]],
                split=inst["split"], signer_id=inst.get("signer_id"),
                frame_start=inst.get("frame_start"), frame_end=inst.get("frame_end"),
                path=str(Path(args.videos) / f"{vid}.mp4"),
                out=str(config.LANDMARK_DIR / f"{vid}.npz"),
                overwrite=args.overwrite, preview_dir=args.preview_dir))
    return jobs


def check_against_nslt(jobs, wlasl_json, n):
    """WLASL ships nslt_<n>.json listing the official subset's videos. If it is there, confirm
    our vocab picks the same videos, so 'WLASL100' means the same thing as in the papers."""
    p = Path(wlasl_json).parent / f"nslt_{n}.json"
    if not p.exists():
        print(f"(no {p.name} next to the json; skipping subset cross-check)")
        return
    official = set(json.load(open(p)))
    ours = {j["video_id"] for j in jobs}
    print(f"subset check vs {p.name}: ours={len(ours)} official={len(official)} "
          f"overlap={len(ours & official)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wlasl-json", default=str(config.WLASL_JSON),
                    help="default: data/wlasl/WLASL_v0.3.json (where scripts/get_wlasl100.py puts it)")
    ap.add_argument("--videos", default=str(config.VIDEO_DIR),
                    help="folder with <video_id>.mp4; default: data/wlasl/videos")
    ap.add_argument("--write-vocab", type=int, metavar="N",
                    help="write the first N glosses of the json to configs/vocab.txt and exit. "
                         "WLASL orders glosses so the first 100 ARE the WLASL100 subset.")
    ap.add_argument("--limit", type=int, help="only process the first N clips that exist on disk")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--preview", action="store_true", help="save one annotated PNG per clip to data/preview/")
    args = ap.parse_args()

    meta = json.load(open(args.wlasl_json))
    if args.write_vocab:
        config.VOCAB_PATH.parent.mkdir(parents=True, exist_ok=True)
        config.VOCAB_PATH.write_text("\n".join(e["gloss"] for e in meta[:args.write_vocab]) + "\n")
        print(f"wrote {args.write_vocab} glosses to {config.VOCAB_PATH}")
        return

    # Check here, not in the workers: an exception inside a Pool initializer makes the
    # pool respawn workers forever instead of failing.
    for p in (config.HAND_MODEL, config.POSE_MODEL):
        if not p.exists():
            sys.exit(f"{p} missing; run: python scripts/get_mediapipe_models.py")
    if not config.VOCAB_PATH.exists():
        sys.exit(f"{config.VOCAB_PATH} missing; run once with --write-vocab 100")
    vocab = [g for g in config.VOCAB_PATH.read_text().splitlines() if g.strip()]
    config.LANDMARK_DIR.mkdir(parents=True, exist_ok=True)
    args.preview_dir = None
    if args.preview:
        args.preview_dir = str(config.ROOT / "data" / "preview")
        Path(args.preview_dir).mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(meta, vocab, args)
    check_against_nslt(jobs, args.wlasl_json, len(vocab))
    present = [j for j in jobs if Path(j["path"]).exists()]
    missing = [j for j in jobs if not Path(j["path"]).exists()]
    print(f"vocab={len(vocab)}  clips in json={len(jobs)}  on disk={len(present)}  missing={len(missing)}")
    if args.limit:
        present = present[:args.limit]

    rows = [{k: j[k] for k in ("video_id", "gloss", "label", "split", "signer_id", "path")} | {"status": "missing_video"}
            for j in missing] if not args.limit else []
    t0 = time.perf_counter()
    with Pool(args.workers, initializer=_init_worker) as pool:
        for i, row in enumerate(pool.imap_unordered(_process, present, chunksize=1), 1):
            rows.append(row)
            if i % 50 == 0 or i == len(present):
                print(f"  {i}/{len(present)} done, {time.perf_counter() - t0:.0f}s elapsed", flush=True)

    df = pd.DataFrame(rows).sort_values(["split", "label", "video_id"])
    manifest = config.MANIFEST_PATH if not args.limit else config.MANIFEST_PATH.with_name("manifest_test.csv")
    df.to_csv(manifest, index=False)
    print(f"\nwrote {manifest}")

    print("\nclips by split and status:")
    print(df.pivot_table(index="split", columns="status", values="video_id", aggfunc="count", fill_value=0))
    ok = df[df.status == "ok"]
    if len(ok):
        print(f"\nper-clip detection (ok clips): frames median={ok.n_frames.median():.0f}  "
              f"any-hand median={ok.frac_any_hand.median():.2f}  pose median={ok.frac_pose.median():.2f}")
        print(f"extraction seconds/clip (incl. model load): median={ok.seconds.median():.2f}  "
              f"max={ok.seconds.max():.2f}  [workers={args.workers}]")
        per_word = ok[ok.split == "train"].groupby("gloss").size()
        if len(per_word):
            print(f"train clips per word: min={per_word.min()}  median={per_word.median():.0f}  "
                  f"words with <5: {(per_word < 5).sum()}")
    if args.limit:
        print("\n" + df.drop(columns=["signer_id", "path"], errors="ignore").to_string(index=False))
    else:
        write_report(df, vocab, args.workers)


def write_report(df, vocab, workers):
    """Usable clips per word and per split -> outputs/clip_counts.csv + extraction_report.md.
    'Usable' = the video decoded and at least one hand was found in at least one frame."""
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ok = df[df.status == "ok"]
    counts = (ok.pivot_table(index="gloss", columns="split", values="video_id",
                             aggfunc="count", fill_value=0)
              .reindex(index=vocab, columns=["train", "val", "test"], fill_value=0))
    counts["unusable"] = df[df.status != "ok"].groupby("gloss").size().reindex(vocab, fill_value=0)
    counts.to_csv(config.OUTPUT_DIR / "clip_counts.csv")

    status = df.pivot_table(index="split", columns="status", values="video_id",
                            aggfunc="count", fill_value=0).reindex(["train", "val", "test"])
    lines = ["# Landmark extraction report", "",
             "Generated by `scripts/extract_all.py`. Do not edit by hand.", "",
             f"- Vocabulary: {len(vocab)} words (`configs/vocab.txt`)",
             f"- Clips listed in the WLASL json for these words: {len(df)}",
             f"- Usable clips (decoded, at least one hand found): {len(ok)}",
             f"- Hand detection confidence {_det_conf()}, at most {_max_frames()} frames per clip", "",
             "## Clips by split and status", "",
             "| split | " + " | ".join(status.columns) + " |",
             "|---|" + "---|" * len(status.columns)]
    lines += [f"| {s} | " + " | ".join(str(int(v)) for v in status.loc[s]) + " |" for s in status.index]
    lines += ["", "## Usable clips per word, by split", ""]
    for s in ("train", "val", "test"):
        c = counts[s]
        lines.append(f"- {s}: total {int(c.sum())}, per word min {int(c.min())}, "
                     f"median {c.median():.0f}, max {int(c.max())}, words with zero clips: {int((c == 0).sum())}")
    if len(ok):
        lines += ["", "## Detection quality on usable clips", "",
                  f"- Frames processed per clip: median {ok.n_frames.median():.0f}, "
                  f"min {int(ok.n_frames.min())}, max {int(ok.n_frames.max())}",
                  f"- Fraction of frames with any hand found: median {ok.frac_any_hand.median():.2f}, "
                  f"mean {ok.frac_any_hand.mean():.2f}",
                  f"- Fraction of frames with pose found: median {ok.frac_pose.median():.2f}, "
                  f"mean {ok.frac_pose.mean():.2f}",
                  f"- Clips with a hand in fewer than half their frames: {int((ok.frac_any_hand < 0.5).sum())}"]
        fresh = ok[ok.status_note != "cached"]
        if len(fresh):
            lines.append(f"- Extraction time per clip (model load included, {workers} parallel workers): "
                         f"median {fresh.seconds.median():.2f} s")
    lines += ["", "Per-word numbers are in `outputs/clip_counts.csv`.", ""]
    (config.OUTPUT_DIR / "extraction_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {config.OUTPUT_DIR / 'clip_counts.csv'} and extraction_report.md")


def _det_conf():
    from src.landmarks import HAND_DET_CONF
    return HAND_DET_CONF


def _max_frames():
    from src.landmarks import MAX_FRAMES
    return MAX_FRAMES


if __name__ == "__main__":   # required on Windows/macOS: worker processes re-import this file
    main()
