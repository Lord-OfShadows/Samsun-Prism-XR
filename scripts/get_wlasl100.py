"""Download the WLASL100 dataset we trained on into data/wlasl/. Run once per machine.

  python scripts/get_wlasl100.py

Fetches, from public sources:
  WLASL_v0.3.json   word list, splits and frame bounds  (official WLASL GitHub repo)
  5 zip files       2,038 clips, 256x256, pre-trimmed   (Hugging Face mirror aipieces/WLASL100)
and unpacks the clips into the flat folder data/wlasl/videos/<video_id>.mp4, which is the
layout every other script expects. About 212 MB to download; files already present are
skipped, so it is safe to rerun.

WLASL is released under the C-UDA licence (computational use only). Do not commit the
videos; data/ is git-ignored.
"""
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

JSON_URL = "https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json"
ZIP_URL = "https://huggingface.co/datasets/aipieces/WLASL100/resolve/main/WLASL100/shard_{:03d}_005.zip"
EXPECTED_CLIPS = 2038


def fetch(url, dest):
    if dest.exists() and dest.stat().st_size > 0:
        print(f"have {dest.name}")
        return
    print(f"downloading {url}", flush=True)
    tmp = dest.with_suffix(dest.suffix + ".part")       # never leave a half file under the real name
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    tmp.replace(dest)


def unpack(zip_path, video_dir):
    """Extract <anything>/<digits>.mp4 members as video_dir/<digits>.mp4, nothing else."""
    n = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = Path(info.filename).name
            if info.is_dir() or not (name.endswith(".mp4") and name[:-4].isdigit()):
                continue
            out = video_dir / name
            if not out.exists():
                with zf.open(info) as src, open(out, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            n += 1
    return n


def main():
    zips = config.WLASL_DIR / "zips"
    zips.mkdir(parents=True, exist_ok=True)
    config.VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    fetch(JSON_URL, config.WLASL_JSON)
    total = 0
    for i in range(1, 6):
        z = zips / f"shard_{i:03d}_005.zip"
        fetch(ZIP_URL.format(i), z)
        total += unpack(z, config.VIDEO_DIR)
    on_disk = len(list(config.VIDEO_DIR.glob("*.mp4")))
    print(f"\n{total} clips in the zips, {on_disk} in {config.VIDEO_DIR}")
    if on_disk < EXPECTED_CLIPS:
        sys.exit(f"expected {EXPECTED_CLIPS} clips; rerun, or check the mirror is still available")
    print("next: python scripts/extract_all.py   (or unzip the shared landmark bundle, see README)")


if __name__ == "__main__":
    main()
