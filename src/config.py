"""Paths and constants shared by every module. Change things here, not in the scripts."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The vocabulary: one gloss per line, line number = class index.
# Retraining on another word list = replace this file and rerun scripts/run_all.py.
VOCAB_PATH = ROOT / "configs" / "vocab.txt"

MP_MODELS_DIR = ROOT / "models" / "mediapipe"
HAND_MODEL = MP_MODELS_DIR / "hand_landmarker.task"
# "lite" pose model: we only need shoulders/elbows/wrists, and lite is the smallest and
# fastest of the three pose variants, which matters for both the size and latency budgets.
POSE_MODEL = MP_MODELS_DIR / "pose_landmarker_lite.task"

WLASL_DIR = ROOT / "data" / "wlasl"           # scripts/get_wlasl100.py downloads into here
WLASL_JSON = WLASL_DIR / "WLASL_v0.3.json"
VIDEO_DIR = WLASL_DIR / "videos"              # <video_id>.mp4
LANDMARK_DIR = ROOT / "data" / "landmarks"   # one .npz per clip
MANIFEST_PATH = ROOT / "data" / "manifest.csv"
ROBUSTNESS_DIR = ROOT / "data" / "robustness"  # cached landmarks of perturbed test clips
# Probability that a training clip gets box_occlusion() (src/augment.py). 0.0 = the
# original 19 Nov model. Set to the value the frozen model was trained with.
OCCLUSION_AUG_P = 0.0

MODELS_DIR = ROOT / "models"
FLOAT_MODEL = MODELS_DIR / "echogesture_float.keras"   # what train.py saves
# The two environment variables exist for trying out another model without touching the
# shipped one or its reported numbers, e.g.
#   ECHOGESTURE_INT8_MODEL=other.tflite ECHOGESTURE_OUTPUT_DIR=outputs_other python -m src.benchmark
INT8_MODEL = Path(os.environ.get("ECHOGESTURE_INT8_MODEL", MODELS_DIR / "echogesture_int8.tflite"))  # what predict() loads
OUTPUT_DIR = Path(os.environ.get("ECHOGESTURE_OUTPUT_DIR", ROOT / "outputs"))

SEQ_LEN = 64        # frames the classifier sees (PRD)
FEATURE_CLIP = 3.0  # every feature is clipped to [-3, 3]: gives INT8 a fixed, known input range

SIZE_BUDGET_MB = 20.0     # the brief says 20 in one place and 25 in another; we design for 20
LATENCY_BUDGET_MS = 50.0
# Which measured latency goes in the single latency_p95 row of results.csv:
# "end_to_end" (whole clip incl. landmarks, the strictest reading), "per_frame" or "classifier".
# All three are always measured and written to outputs/benchmark.json and the notes column.
# "classifier": the official matrix says "inference time < 50 ms". Per-frame and whole-clip
# p95 are still measured and shown in the README (results_notes.csv).
LATENCY_METRIC_FOR_RESULTS = "classifier"

# Live demo caption behaviour
CAPTION_THRESHOLD = 0.6   # show a word only above this confidence
CAPTION_HOLD_S = 1.5      # and keep it on screen at least this long
SPEAK_CAPTIONS = True     # speak each new caption offline (demo/speech.py); --no-speak turns it off


def load_vocab(path=None):
    """Class index -> word. The ONLY place the word list is read."""
    p = Path(path) if path else VOCAB_PATH
    return [g.strip() for g in p.read_text(encoding="utf-8").splitlines() if g.strip()]
