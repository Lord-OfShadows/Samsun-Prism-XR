"""Download the two MediaPipe model files into models/mediapipe/ and print their sizes.

The sizes printed here are the real numbers for the size budget (the PRD asks whether
third-party models count, so we record them from day one).
"""
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `src` imports from scripts/
from src.config import HAND_MODEL, POSE_MODEL

BASE = "https://storage.googleapis.com/mediapipe-models"
URLS = {
    HAND_MODEL: f"{BASE}/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    POSE_MODEL: f"{BASE}/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
}

if __name__ == "__main__":
    for dest, url in URLS.items():
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, dest)
        print(f"{dest.name}: {dest.stat().st_size} bytes ({dest.stat().st_size / 2**20:.2f} MiB)")
