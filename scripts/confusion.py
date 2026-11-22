"""Which words does the model mix up? Reads predictions, writes the most confused pairs.

  python scripts/confusion.py            # top 10 pairs
  python scripts/confusion.py --top 20

Inputs:  outputs/predictions.jsonl (predicted word per test clip, from make_predictions.py)
         data/manifest.csv         (true word per clip)
Outputs: outputs/confusion.csv     (the most confused word pairs)
         outputs/confusion.json    (the same pairs plus per-word accuracy, for the README)

A pair is unordered: "A read as B" and "B read as A" both count towards {A, B}, because
for the jury question ("which signs does it confuse?") the direction matters less than
the fact that the two look alike to the model. Both directions are still reported.
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402


def load_truth(manifest_path, split="test"):
    """clip_id -> true word, for one split of the manifest."""
    with open(manifest_path, encoding="utf-8") as f:
        return {r["video_id"]: r["gloss"] for r in csv.DictReader(f) if r["split"] == split}


def load_predictions(path):
    """clip_id -> predicted word."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                # predict() returns upper-case words; vocab.txt is lower-case
                out[r["clip_id"]] = r["prediction"].lower() if r["prediction"] else None
    return out


def confused_pairs(truth, preds, top=10):
    """Rank unordered word pairs by how often one was predicted for the other.

    Returns a list of dicts sorted by total errors (ties broken alphabetically, so the
    output is stable), with the count in each direction.
    """
    directed = Counter((truth[c], preds[c]) for c in preds if c in truth and preds[c] != truth[c])
    pairs = Counter()
    for (t, p), n in directed.items():
        pairs[tuple(sorted((t, p)))] += n
    ranked = sorted(pairs.items(), key=lambda kv: (-kv[1], kv[0]))[:top]
    return [{"word_a": a, "word_b": b, "a_read_as_b": directed[(a, b)],
             "b_read_as_a": directed[(b, a)], "total": n} for (a, b), n in ranked]


def per_word_accuracy(truth, preds):
    """word -> (correct, clips) over the clips that have a prediction."""
    clips, correct = Counter(), Counter()
    for c, word in truth.items():
        if c in preds:
            clips[word] += 1
            correct[word] += preds[c] == word
    return {w: (correct[w], clips[w]) for w in clips}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--predictions", default=str(config.OUTPUT_DIR / "predictions.jsonl"))
    ap.add_argument("--manifest", default=str(config.MANIFEST_PATH))
    args = ap.parse_args()

    truth, preds = load_truth(args.manifest), load_predictions(args.predictions)
    missing = sorted(set(truth) - set(preds))
    if missing:
        print(f"warning: {len(missing)} test clips have no prediction, e.g. {missing[:3]}")
    scored = [c for c in preds if c in truth]
    errors = sum(preds[c] != truth[c] for c in scored)
    pairs = confused_pairs(truth, preds, args.top)

    out_csv = config.OUTPUT_DIR / "confusion.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["rank", "word_a", "word_b", "a_read_as_b", "b_read_as_a", "total"])
        w.writeheader()
        for i, p in enumerate(pairs, 1):
            w.writerow({"rank": i, **p})

    acc = per_word_accuracy(truth, preds)
    worst = sorted(acc.items(), key=lambda kv: (kv[1][0] / kv[1][1], -kv[1][1], kv[0]))[:10]
    # Words the model guesses when it is wrong. One word soaking up many errors from
    # different signs points at a "default" guess, not at one look-alike pair.
    sinks = Counter(preds[c] for c in scored if preds[c] != truth[c])
    summary = {
        "clips_scored": len(scored),
        "errors": errors,
        "errors_in_top_pairs": sum(p["total"] for p in pairs),
        "distinct_error_pairs": len(confused_pairs(truth, preds, top=10**9)),
        "pairs": pairs,
        "worst_words": [{"word": w, "correct": c, "clips": n} for w, (c, n) in worst],
        "most_frequent_wrong_guesses": [{"word": w, "times": n}
                                        for w, n in sorted(sinks.items(), key=lambda kv: (-kv[1], kv[0]))[:5]],
        "words_never_wrong": sum(c == n for c, n in acc.values()),
        "words_scored": len(acc),
    }
    (config.OUTPUT_DIR / "confusion.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    print(f"{errors} errors in {len(scored)} test clips; the top {len(pairs)} pairs account for "
          f"{summary['errors_in_top_pairs']}")
    for i, p in enumerate(pairs, 1):
        print(f"{i:2d}. {p['word_a']:>12} <-> {p['word_b']:<12} {p['total']} "
              f"({p['a_read_as_b']} + {p['b_read_as_a']})")
    print(f"wrote {out_csv} and confusion.json")


if __name__ == "__main__":
    main()
