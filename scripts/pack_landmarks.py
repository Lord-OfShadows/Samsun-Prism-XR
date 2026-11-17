"""Bundle the cached landmarks into one zip to hand to teammates, or unpack one.

  python scripts/pack_landmarks.py            # writes data/share/echogesture_wlasl100_landmarks.zip
  python scripts/pack_landmarks.py --unpack path/to/echogesture_wlasl100_landmarks.zip

The zip holds data/manifest.csv and data/landmarks/*.npz: everything `python -m src.train`
needs, and nothing else. A teammate who unpacks it can train immediately, without the
videos and without the hour of MediaPipe extraction. (The videos are still needed for
predictions, the benchmark and the robustness test: scripts/get_wlasl100.py fetches them.)

The landmarks are derived from WLASL, which is under the C-UDA licence (computational
use only). Share the zip with the team; do not post it publicly without checking that.
"""
import argparse
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

BUNDLE = config.ROOT / "data" / "share" / "echogesture_wlasl100_landmarks.zip"
NOTE = """EchoGesture landmark bundle
Unpack with:  python scripts/pack_landmarks.py --unpack <this zip>
Contents:     data/manifest.csv, data/landmarks/<video_id>.npz
Each .npz:    hands (T,2,21,3), pose (T,33,4) in pixels, NaN = not detected; see src/landmarks.py
Source:       WLASL (Li et al., 2020), 100-word subset, MediaPipe hand + pose landmarks.
Licence:      WLASL is under the C-UDA licence: computational use only.
"""


def pack():
    files = sorted(config.LANDMARK_DIR.glob("*.npz"))
    if not files or not config.MANIFEST_PATH.exists():
        sys.exit("nothing to pack: run scripts/extract_all.py first")
    BUNDLE.parent.mkdir(parents=True, exist_ok=True)
    # ZIP_STORED: the .npz files are already compressed, deflating again only costs time
    with zipfile.ZipFile(BUNDLE, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("README_DATA.txt", NOTE)
        zf.write(config.MANIFEST_PATH, "data/manifest.csv", compress_type=zipfile.ZIP_DEFLATED)
        for f in files:
            zf.write(f, f"data/landmarks/{f.name}")
    print(f"wrote {BUNDLE}: {len(files)} clips, {BUNDLE.stat().st_size / 1e6:.1f} MB")


def unpack(path):
    n = 0
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            name = info.filename
            ok = name == "data/manifest.csv" or (
                name.startswith("data/landmarks/") and name.endswith(".npz")
                and "/" not in name[len("data/landmarks/"):])
            if not ok:                      # only the two expected kinds of file are written
                continue
            out = config.ROOT / name
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(zf.read(info))
            n += 1
    print(f"unpacked {n} files into {config.ROOT / 'data'}; next: python -m src.train")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--unpack", metavar="ZIP")
    a = ap.parse_args()
    unpack(a.unpack) if a.unpack else pack()
