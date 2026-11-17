"""Paths and constants shared by every module. Change things here, not in the scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The vocabulary: one gloss per line, line number = class index.
# Retraining on another word list = replace this file and rerun extraction + training.
VOCAB_PATH = ROOT / "configs" / "vocab.txt"

MP_MODELS_DIR = ROOT / "models" / "mediapipe"
HAND_MODEL = MP_MODELS_DIR / "hand_landmarker.task"
# "lite" pose model: we only need shoulders/elbows/wrists, and lite is the smallest and
# fastest of the three pose variants, which matters for both the size and latency budgets.
POSE_MODEL = MP_MODELS_DIR / "pose_landmarker_lite.task"

LANDMARK_DIR = ROOT / "data" / "landmarks"   # one .npz per clip
MANIFEST_PATH = ROOT / "data" / "manifest.csv"

SEQ_LEN = 64  # frames the classifier sees (PRD). Resampling happens in normalize.py, not here.
