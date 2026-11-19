"""The confusion analysis: pairs are unordered, counted in both directions, and ranked stably."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from confusion import confused_pairs, per_word_accuracy  # noqa: E402

TRUTH = {"1": "dog", "2": "later", "3": "dog", "4": "no", "5": "right", "6": "book", "7": "book"}
PREDS = {"1": "later", "2": "dog", "3": "dog", "4": "right", "5": "yes", "6": "book", "7": "book"}


def test_both_directions_count_towards_one_pair():
    top = confused_pairs(TRUTH, PREDS)[0]
    assert (top["word_a"], top["word_b"]) == ("dog", "later")
    assert top["a_read_as_b"] == 1 and top["b_read_as_a"] == 1 and top["total"] == 2


def test_correct_predictions_are_not_pairs_and_ties_are_alphabetical():
    pairs = confused_pairs(TRUTH, PREDS)
    assert [(p["word_a"], p["word_b"]) for p in pairs] == [("dog", "later"), ("no", "right"), ("right", "yes")]
    assert all(p["word_a"] != p["word_b"] for p in pairs)
    assert len(confused_pairs(TRUTH, PREDS, top=1)) == 1


def test_clips_missing_from_either_side_are_ignored():
    pairs = confused_pairs({**TRUTH, "99": "dog"}, {**PREDS, "100": "cat"})
    assert sum(p["total"] for p in pairs) == 4


def test_per_word_accuracy():
    acc = per_word_accuracy(TRUTH, PREDS)
    assert acc["dog"] == (1, 2) and acc["book"] == (2, 2) and acc["right"] == (0, 1)
