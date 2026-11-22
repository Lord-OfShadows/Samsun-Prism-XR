"""The function the evaluator runs: one video clip in, one English word out, fully offline.

    from predict import predict
    result = predict("clip.mp4")        # a dict, json-serialisable

    python predict.py clip.mp4          # same thing from the command line, prints JSON

Stages (each timed separately): decode video -> MediaPipe landmarks -> normalise ->
INT8 classifier. Everything loads from local files in models/; nothing touches the network.
The models are loaded once, on the first call, and reused.

The output schema is produced by format_output() and nowhere else. The organisers' exact
format (docs/BRIEF_FORMATS.md) was not available when this was written, so if their
schema differs, format_output() is the one function to change. See docs/ASSUMPTIONS.md.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import config  # noqa: E402

_pipeline = None   # (extractor, classifier, vocab), built on first use


def _load():
    global _pipeline
    if _pipeline is None:
        from src.classifier import Int8Classifier
        from src.landmarks import LandmarkExtractor
        vocab = config.load_vocab()
        clf = Int8Classifier(config.INT8_MODEL)
        if clf.n_classes != len(vocab):
            raise ValueError(f"model has {clf.n_classes} outputs but configs/vocab.txt has "
                             f"{len(vocab)} words; retrain or restore the matching vocab")
        _pipeline = (LandmarkExtractor(config.HAND_MODEL, config.POSE_MODEL), clf, vocab)
    return _pipeline


def format_output(clip_id, probs, vocab, latency_ms, n_frames=None):
    """THE output schema, exactly the brief's example:
    {"clip_id": "test_0412", "prediction": "HAPPY", "top5": ["HAPPY", ...],
     "confidence": 0.81, "latency_ms": 23.4}
    Words are upper-case because the brief's example is. latency_ms is the total for this
    call; the per-stage split is measured by src/benchmark.py, not returned here, so the
    evaluator gets no keys it did not ask for. n_frames is accepted but not returned."""
    order = np.argsort(-probs)[:5]
    return {
        "clip_id": clip_id,
        "prediction": vocab[int(order[0])].upper(),
        "top5": [vocab[int(i)].upper() for i in order],
        "confidence": round(float(probs[order[0]]), 4),
        "latency_ms": round(float(latency_ms["total"]), 2),
    }


def run_stages(video_path):
    """Run the pipeline on one clip. Returns (probs, latency_ms dict, frames processed)."""
    from src.landmarks import read_frames, shrink
    from src.normalize import normalize

    extractor, clf, _ = _load()
    t0 = time.perf_counter()
    frames, _, fps, _ = read_frames(video_path)
    if not frames:
        raise IOError(f"no frames decoded from {video_path}")
    frames = shrink(frames)
    t1 = time.perf_counter()
    hands, pose = extractor.extract_frames(frames, fps)
    t2 = time.perf_counter()
    feats = normalize(hands, pose, frames[0].shape[1])
    t3 = time.perf_counter()
    probs = clf.probs(feats)
    t4 = time.perf_counter()
    ms = {"decode": (t1 - t0) * 1e3, "landmarks": (t2 - t1) * 1e3, "normalize": (t3 - t2) * 1e3,
          "classifier": (t4 - t3) * 1e3, "total": (t4 - t0) * 1e3}
    return probs, ms, len(frames)


def predict(video_path):
    """Recognise the sign in one video clip. Returns the dict built by format_output()."""
    _, _, vocab = _load()
    probs, ms, n_frames = run_stages(video_path)
    return format_output(Path(video_path).stem, probs, vocab, ms, n_frames)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python predict.py <video file>")
    print(json.dumps(predict(sys.argv[1]), indent=2))
