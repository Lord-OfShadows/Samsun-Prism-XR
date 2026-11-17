"""The JSON contract of predict(), the results.csv scoring, and the caption logic."""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "demo"))

from predict import format_output  # noqa: E402

VOCAB = [f"word{i}" for i in range(100)]
MS = {"decode": 1.0, "landmarks": 2.0, "normalize": 0.5, "classifier": 0.25, "total": 3.75}


def _probs(top=7):
    p = np.full(100, 0.001, np.float32)
    p[top], p[3], p[50] = 0.6, 0.2, 0.1
    return p


def test_schema_and_types():
    out = format_output("00618", _probs(), VOCAB, MS, 64)
    assert set(out) == {"clip_id", "prediction", "confidence", "top5", "latency_ms", "frames"}
    assert out["clip_id"] == "00618" and out["prediction"] == "word7" and out["frames"] == 64
    assert isinstance(out["confidence"], float) and abs(out["confidence"] - 0.6) < 1e-4
    assert set(out["latency_ms"]) == {"decode", "landmarks", "normalize", "classifier", "total"}


def test_top5_is_sorted_and_consistent_with_prediction():
    out = format_output("x", _probs(), VOCAB, MS, 64)
    words = [t["word"] for t in out["top5"]]
    confs = [t["confidence"] for t in out["top5"]]
    assert len(words) == 5 and len(set(words)) == 5
    assert words[:3] == ["word7", "word3", "word50"]
    assert confs == sorted(confs, reverse=True)
    assert out["top5"][0] == {"word": out["prediction"], "confidence": out["confidence"]}


def test_output_is_plain_json():
    out = format_output("x", _probs(), VOCAB, MS, 64)
    assert json.loads(json.dumps(out)) == out       # no numpy types leak through


def test_scoring_counts_missing_and_failed_clips_as_wrong():
    from make_results import score_predictions
    lines = [
        {"clip_id": "a", "prediction": "word1", "top5": [{"word": "word1"}]},              # right
        {"clip_id": "b", "prediction": "word9", "top5": [{"word": "word9"}, {"word": "word2"}]},  # top5 only
        {"clip_id": "c", "prediction": None, "top5": []},                                  # failed clip
    ]
    labels = {"a": 1, "b": 2, "c": 3, "d": 4}                                              # d is absent
    top1, top5, n = score_predictions(lines, labels, VOCAB)
    assert (top1, top5, n) == (0.25, 0.5, 4)


def test_results_csv_has_six_rows_and_never_invents_phone_numbers(tmp_path, monkeypatch):
    import make_results
    monkeypatch.setattr(make_results.config, "OUTPUT_DIR", tmp_path)       # no inputs present
    monkeypatch.setattr(make_results.config, "MANIFEST_PATH", tmp_path / "none.csv")
    make_results.main()
    rows = list(csv.DictReader(open(tmp_path / "results.csv", encoding="utf-8")))
    assert [r["metric"] for r in rows] == ["top1_accuracy", "latency_p95", "model_size",
                                           "accuracy_drop_robustness", "avg_power", "thermal_drift_10min"]
    assert all(r["value"] == "not measured" and r["pass"] == "not measured" for r in rows)


def test_caption_needs_confidence_and_holds():
    from live import CaptionStabilizer
    c = CaptionStabilizer(threshold=0.6, hold_s=1.5)
    assert c.update("book", 0.4, 0.0) is None             # below threshold: nothing shown
    assert c.update("book", 0.8, 0.3) == "book"           # confident: shown
    assert c.update("drink", 0.9, 0.6) == "book"          # too soon to switch: held
    assert c.update(None, 0.0, 1.0) == "book"             # a gap does not blank it immediately
    assert c.update("drink", 0.9, 2.0) == "drink"         # hold elapsed: may switch
    assert c.update(None, 0.0, 2.5) == "drink"
    assert c.update(None, 0.0, 4.0) is None               # nothing confident for a while: clear
