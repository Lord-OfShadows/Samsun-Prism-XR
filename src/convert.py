"""Keras float model -> full-integer INT8 TFLite, then check what the conversion cost.

In:  models/echogesture_float.keras, cached training landmarks (the representative set)
Out: models/echogesture_int8.tflite
     outputs/quantization.json   file sizes, float vs INT8 accuracy on val and test

  python -m src.convert

Post-training quantisation needs to know the range of values flowing through every layer.
The converter learns those ranges by running a "representative dataset" through the float
model; we give it real, un-augmented training sequences so the ranges match real inputs.
"Full-integer" means weights, activations, input and output are all int8: no float maths
is left in the model, which is what lets it run on integer-only accelerators.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from src import config

N_REPRESENTATIVE = 300


def to_int8_tflite(model, representative_x):
    """Return the INT8 flatbuffer (bytes) for a Keras model. Batch size is fixed to 1."""
    import tempfile

    import tensorflow as tf

    # Keras 3 -> SavedModel -> TFLite. Going through a SavedModel freezes the weights into
    # the graph; converting the live Keras object directly left READ_VARIABLE ops that the
    # calibrator could not run (TF 2.21 / Keras 3.15).
    tmp = tempfile.mkdtemp(prefix="echogesture_savedmodel_")
    spec = tf.TensorSpec([1] + list(model.input_shape[1:]), tf.float32, name="landmarks")
    model.export(tmp, format="tf_saved_model", input_signature=[spec], verbose=False)
    converter = tf.lite.TFLiteConverter.from_saved_model(tmp)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: (
        [representative_x[i:i + 1].astype(np.float32)] for i in range(len(representative_x)))
    # INT8 builtins only: conversion FAILS if any op has no integer kernel, instead of
    # silently leaving float ops in the graph.
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def float_tensors(tflite_path):
    """Names of float tensors left in the model. Empty list = truly full-integer."""
    from ai_edge_litert.interpreter import Interpreter
    interp = Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    return [t["name"] for t in interp.get_tensor_details()
            if np.issubdtype(t["dtype"], np.floating)]


def topk(probs, labels, k):
    return float((np.argsort(-probs, axis=1)[:, :k] == labels[:, None]).any(axis=1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--float-model", default=str(config.FLOAT_MODEL))
    ap.add_argument("--out", default=str(config.INT8_MODEL))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import keras
    import src.model  # noqa: F401  registers the custom attention layer for loading
    from src.classifier import Int8Classifier
    from src.data import features, load_manifest, load_split

    model = keras.saving.load_model(args.float_model)
    manifest = load_manifest()
    train_raw, _, _ = load_split("train", manifest)
    pick = np.random.default_rng(args.seed).permutation(len(train_raw))[:N_REPRESENTATIVE]
    rep_x = features([train_raw[i] for i in pick])

    blob = to_int8_tflite(model, rep_x)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_bytes(blob)
    leftover = float_tensors(args.out)
    clf = Int8Classifier(args.out)
    print(f"wrote {args.out}: {len(blob):,} bytes ({len(blob) / 1e6:.2f} MB), "
          f"float tensors left: {len(leftover)}")
    print(f"input  scale {clf.in_scale:.6f} zero_point {clf.in_zero}   "
          f"output scale {clf.out_scale:.6f} zero_point {clf.out_zero}")

    report = {"float_model_bytes": Path(args.float_model).stat().st_size,
              "int8_model_bytes": len(blob), "representative_clips": int(len(rep_x)),
              "float_tensors_left": len(leftover), "parameters": int(model.count_params()),
              "input_scale": float(clf.in_scale), "input_zero_point": int(clf.in_zero)}
    for split in ("val", "test"):
        raws, y, _ = load_split(split, manifest, usable_only=False)
        x = features(raws)
        pf = model.predict(x, batch_size=256, verbose=0)
        pq = clf.probs_batch(x)
        report[split] = {"clips": len(y),
                         "float_top1": topk(pf, y, 1), "int8_top1": topk(pq, y, 1),
                         "float_top5": topk(pf, y, 5), "int8_top5": topk(pq, y, 5),
                         "same_top1_prediction": float((pf.argmax(1) == pq.argmax(1)).mean())}
        r = report[split]
        print(f"{split:4s} ({r['clips']} clips)  top1 float {r['float_top1']:.4f} -> int8 {r['int8_top1']:.4f}   "
              f"top5 float {r['float_top5']:.4f} -> int8 {r['int8_top5']:.4f}   "
              f"agree {r['same_top1_prediction']:.3f}")
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (config.OUTPUT_DIR / "quantization.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
