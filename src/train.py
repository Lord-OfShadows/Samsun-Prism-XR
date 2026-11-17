"""Train the classifier on cached landmarks.

In:  data/manifest.csv + data/landmarks/*.npz (from scripts/extract_all.py), configs/vocab.txt
Out: models/echogesture_float.keras   best weights by validation accuracy
     outputs/train_log.json           settings, per-epoch history, final float accuracies

  python -m src.train                       # default: conv + attention
  python -m src.train --arch conv           # the PRD's fallback architecture

Every epoch re-augments every training clip (src/augment.py), so the model never sees the
same example twice. Validation and test clips are never augmented. The test split is only
read once, at the very end, to report a number; it plays no part in choosing the model.
"""
import argparse
import json
import time

import keras
import numpy as np

from src import config
from src.augment import augment
from src.data import features, load_manifest, load_split
from src.model import build_model
from src.normalize import normalize


def augmented_features(raws, rng):
    return np.stack([normalize(*augment(r["hands"], r["pose"], r["width"], r["height"], rng), r["width"])
                     for r in raws])


def topk(probs, labels, k):
    return float((np.argsort(-probs, axis=1)[:, :k] == labels[:, None]).any(axis=1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="conv_attn", choices=["conv_attn", "conv"])
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--patience", type=int, default=60, help="stop after this many epochs without a better val accuracy")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.02)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(config.FLOAT_MODEL))
    args = ap.parse_args()

    keras.utils.set_random_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    vocab = config.load_vocab()
    n_classes = len(vocab)

    manifest = load_manifest()
    train_raw, y_train, _ = load_split("train", manifest)
    val_raw, y_val, _ = load_split("val", manifest)
    x_val = features(val_raw)
    print(f"train clips {len(train_raw)}  val clips {len(val_raw)}  classes {n_classes}")

    model = build_model(n_classes, arch=args.arch)
    steps = int(np.ceil(len(train_raw) / args.batch))
    schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=args.lr / 20, warmup_target=args.lr,
        warmup_steps=10 * steps, decay_steps=(args.epochs - 10) * steps, alpha=0.01)
    model.compile(
        optimizer=keras.optimizers.AdamW(schedule, weight_decay=args.weight_decay, clipnorm=1.0),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=args.label_smoothing),
        metrics=["accuracy"])
    n_params = model.count_params()
    print(f"{model.name}: {n_params:,} parameters")

    y_train_1h = keras.utils.to_categorical(y_train, n_classes)
    best = {"val_top1": -1.0, "epoch": 0}
    best_weights, history, t0 = None, [], time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        x_train = augmented_features(train_raw, rng)
        h = model.fit(x_train, y_train_1h, batch_size=args.batch, shuffle=True,
                      initial_epoch=epoch - 1, epochs=epoch, verbose=0).history
        pv = model.predict(x_val, batch_size=256, verbose=0)
        row = {"epoch": epoch, "train_loss": round(h["loss"][0], 4),
               "train_acc": round(h["accuracy"][0], 4),
               "val_top1": round(topk(pv, y_val, 1), 4), "val_top5": round(topk(pv, y_val, 5), 4)}
        history.append(row)
        if row["val_top1"] > best["val_top1"]:
            best, best_weights = row, model.get_weights()
        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}  loss {row['train_loss']:.3f}  train {row['train_acc']:.3f}  "
                  f"val top1 {row['val_top1']:.3f} top5 {row['val_top5']:.3f}  "
                  f"(best {best['val_top1']:.3f} @ {best['epoch']})  {time.perf_counter() - t0:.0f}s", flush=True)
        if epoch - best["epoch"] >= args.patience:
            print(f"early stop at epoch {epoch}")
            break

    model.set_weights(best_weights)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(args.out)

    # Final numbers. usable_only=False: a test clip with no hands found still counts.
    test_raw, y_test, _ = load_split("test", manifest, usable_only=False)
    pt = model.predict(features(test_raw), batch_size=256, verbose=0)
    pv = model.predict(x_val, batch_size=256, verbose=0)
    result = {
        "arch": args.arch, "parameters": int(n_params), "args": vars(args),
        "train_clips": len(train_raw), "val_clips": len(val_raw), "test_clips": len(test_raw),
        "best_epoch": best["epoch"], "epochs_run": len(history),
        "float_val_top1": topk(pv, y_val, 1), "float_val_top5": topk(pv, y_val, 5),
        "float_test_top1": topk(pt, y_test, 1), "float_test_top5": topk(pt, y_test, 5),
        "train_seconds": round(time.perf_counter() - t0, 1), "history": history,
    }
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (config.OUTPUT_DIR / "train_log.json").write_text(json.dumps(result, indent=1))
    print(f"\nsaved {args.out}")
    print(f"float model  val top1 {result['float_val_top1']:.4f} top5 {result['float_val_top5']:.4f}  |  "
          f"test top1 {result['float_test_top1']:.4f} top5 {result['float_test_top5']:.4f}")


if __name__ == "__main__":
    main()
