"""Read the manifest and the cached landmark files. Shared by train, convert, and the scripts.

The manifest (data/manifest.csv, written by scripts/extract_all.py) has one row per clip:
video_id, gloss, label, split, status, path. video_id must stay a STRING: WLASL ids have
leading zeros ("00618") that pandas would otherwise strip.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.normalize import normalize


def load_manifest(path=None):
    p = path or config.MANIFEST_PATH
    if not p.exists():
        raise FileNotFoundError(f"{p} missing; run scripts/extract_all.py first (see README)")
    df = pd.read_csv(p, dtype={"video_id": str})
    df["path"] = [str(_find_video(v, raw)) for v, raw in zip(df.video_id, df.path)]
    return df


def _find_video(video_id, recorded):
    """Where is this clip's video on THIS machine? The manifest may have been made on a
    teammate's computer (other drive, other slashes), so try the recorded path, then the
    same path under this repo, then the standard folder data/wlasl/videos/."""
    rel = Path(str(recorded).replace("\\", "/"))
    for cand in (rel, config.ROOT / rel, config.VIDEO_DIR / f"{video_id}.mp4"):
        if cand.exists():
            return cand
    return config.VIDEO_DIR / f"{video_id}.mp4"


def load_raw(video_id, landmark_dir=None):
    """One cached clip -> dict(hands, pose, width, height)."""
    with np.load((landmark_dir or config.LANDMARK_DIR) / f"{video_id}.npz") as d:
        return {"hands": d["hands"], "pose": d["pose"],
                "width": float(d["width"]), "height": float(d["height"])}


def load_split(split, manifest=None, usable_only=True):
    """Return (list of raw clips, labels array, video_ids) for 'train', 'val' or 'test'.

    usable_only drops clips where no hand was ever found. Training wants that; scoring the
    test split must NOT (an unusable test clip still counts as a miss), so the evaluation
    scripts pass usable_only=False and such clips are fed as all-zero features.
    """
    df = manifest if manifest is not None else load_manifest()
    df = df[df.split == split]
    df = df[df.status == "ok"] if usable_only else df[df.status.isin(["ok", "no_hands"])]
    raws = [load_raw(v) for v in df.video_id]
    return raws, df.label.to_numpy(np.int64), df.video_id.tolist()


def features(raws):
    """Normalise a list of raw clips -> float32 (N, 64, 227). No augmentation."""
    return np.stack([normalize(r["hands"], r["pose"], r["width"]) for r in raws])
