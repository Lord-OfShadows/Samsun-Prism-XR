"""box_occlusion(): the whole-clip occlusion augmentation."""
import numpy as np

from src.augment import augment, box_occlusion
from tests.conftest import W, make_clip


def _hands_at(x, y, T=10):
    h = np.full((T, 2, 21, 3), np.nan, np.float32)
    h[:, 0, :, 0], h[:, 0, :, 1], h[:, 0, :, 2] = x, y, 0.0     # only the left hand present
    return h


def test_box_covering_the_frame_removes_every_hand():
    out = box_occlusion(_hands_at(128, 128), W, W, np.random.default_rng(0), min_frac=1.0, max_frac=1.0)
    assert np.isnan(out).all()


def test_hand_outside_a_small_box_is_kept():
    h = _hands_at(3, 3)        # top-left corner; the box centre is never placed there
    out = box_occlusion(h, W, W, np.random.default_rng(0), min_frac=0.05, max_frac=0.05)
    np.testing.assert_array_equal(out, h)


def test_input_is_not_modified():
    h = _hands_at(128, 128)
    box_occlusion(h, W, W, np.random.default_rng(0), min_frac=1.0, max_frac=1.0)
    assert np.isfinite(h[:, 0]).all()


def test_occlusion_off_reproduces_original_augmentation():
    hands, pose = make_clip()
    a = augment(hands, pose, W, W, np.random.default_rng(5))
    b = augment(hands, pose, W, W, np.random.default_rng(5), occlusion_p=0.0)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)
