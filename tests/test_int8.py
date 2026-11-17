"""INT8 input scaling and the conversion itself, on a tiny throwaway model (no dataset)."""
import numpy as np
import pytest

from src.classifier import dequantize, quantize
from src.config import FEATURE_CLIP, SEQ_LEN
from src.normalize import N_FEATURES


def test_quantize_formula_and_saturation():
    scale, zp = 0.02, -3
    x = np.array([0.0, 0.02, -0.02, 0.5, 100.0, -100.0], np.float32)
    q = quantize(x, scale, zp)
    assert q.dtype == np.int8
    assert q.tolist() == [-3, -2, -4, 22, 127, -128]       # last two saturate, never wrap


def test_roundtrip_error_is_at_most_half_a_step():
    scale, zp = 2 * FEATURE_CLIP / 255, -1                 # the scale a [-3, 3] input gets
    x = np.linspace(-FEATURE_CLIP, FEATURE_CLIP, 1001, dtype=np.float32)
    err = np.abs(dequantize(quantize(x, scale, zp), scale, zp) - x)
    assert err.max() <= scale / 2 + 1e-6


@pytest.fixture(scope="module")
def tiny_int8(tmp_path_factory):
    import keras

    from src.classifier import Int8Classifier
    from src.convert import float_tensors, to_int8_tflite
    from src.model import build_model

    keras.utils.set_random_seed(0)
    rng = np.random.default_rng(0)
    x = np.clip(rng.normal(0, 1, (48, SEQ_LEN, N_FEATURES)), -FEATURE_CLIP, FEATURE_CLIP).astype(np.float32)
    x[:, :, -3:] = rng.integers(0, 2, (48, SEQ_LEN, 3))    # mask flags are 0/1, like real inputs
    model = build_model(10, dim=32)                        # same layers as the real one, just narrow
    model.compile("adam", "sparse_categorical_crossentropy")
    model.fit(x, rng.integers(0, 10, 48), epochs=2, verbose=0)   # so BatchNorm has real statistics
    path = tmp_path_factory.mktemp("m") / "tiny_int8.tflite"
    path.write_bytes(to_int8_tflite(model, x))
    return model, Int8Classifier(path), x, float_tensors(path)


def test_conversion_is_full_integer(tiny_int8):
    _, clf, _, leftover = tiny_int8
    assert leftover == []                                  # no float tensor anywhere in the graph
    assert clf.inp["dtype"] == np.int8 and clf.out["dtype"] == np.int8
    assert tuple(clf.inp["shape"]) == (1, SEQ_LEN, N_FEATURES)


def test_input_scale_covers_the_feature_range(tiny_int8):
    _, clf, _, _ = tiny_int8
    lo = dequantize(-128, clf.in_scale, clf.in_zero)
    hi = dequantize(127, clf.in_scale, clf.in_zero)
    assert lo <= -0.9 * FEATURE_CLIP and hi >= 0.9 * FEATURE_CLIP   # [-3, 3] fits without clipping
    assert clf.in_scale < 0.05                                      # and is not wastefully coarse


def test_int8_output_is_a_probability_distribution_close_to_float(tiny_int8):
    model, clf, x, _ = tiny_int8
    pf = model.predict(x[:16], verbose=0)
    pq = clf.probs_batch(x[:16])
    assert pq.shape == pf.shape and (pq >= 0).all()
    assert np.allclose(pq.sum(axis=1), 1, atol=0.05)
    assert np.abs(pq - pf).max() < 0.1


def test_wrong_input_scaling_would_be_caught(tiny_int8):
    """Feeding raw floats cast to int8 (the classic bug) must differ from correct scaling."""
    _, clf, x, _ = tiny_int8
    wrong = x[:1].astype(np.int8)
    right = quantize(x[:1], clf.in_scale, clf.in_zero)
    assert not np.array_equal(wrong, right)
