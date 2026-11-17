"""Normalisation: the fragile part. Wrong here = the model silently learns garbage."""
import numpy as np

from conftest import W, make_clip
from src.augment import augment, mirror
from src.config import FEATURE_CLIP, SEQ_LEN
from src.normalize import (MASK_SLICE, N_FEATURES, N_HAND_BODY_F, N_POSE_F, body_frame, normalize,
                           resample)

LOCAL = slice(N_POSE_F + N_HAND_BODY_F, N_FEATURES - 3)     # the hand-frame block


def test_shape_range_and_no_nan(clip):
    f = normalize(*clip, W)
    assert f.shape == (SEQ_LEN, N_FEATURES) == (64, 227)
    assert f.dtype == np.float32 and np.isfinite(f).all()
    assert f.min() >= -FEATURE_CLIP and f.max() <= FEATURE_CLIP


def test_invariant_to_where_the_signer_stands_and_how_far(clip):
    base = normalize(*clip, W)
    moved = normalize(*make_clip(shift=(25, -12)), W)
    nearer = normalize(*make_clip(zoom=1.4), W)
    assert np.allclose(base, moved, atol=1e-3)
    assert np.allclose(base, nearer, atol=1e-3)


def test_body_frame_is_shoulder_centred_and_shoulder_scaled(clip):
    hands, pose = clip
    centre, scale, ok = body_frame(pose, W)
    assert np.allclose(centre, [128, 100]) and np.isclose(scale, 64) and ok.all()
    f = normalize(hands, pose, W)
    # POSE_KEEP order: nose, L shoulder, R shoulder ... -> shoulders sit at (+0.5, 0), (-0.5, 0)
    assert np.allclose(f[:, 2:6], [0.5, 0, -0.5, 0], atol=1e-4)


def test_hand_frame_is_wrist_centred_and_palm_scaled(clip):
    f = normalize(*clip, W)
    left = f[:, LOCAL][:, :63].reshape(SEQ_LEN, 21, 3)
    assert np.allclose(left[:, 0], 0, atol=1e-5)                          # wrist at origin
    assert np.allclose(np.linalg.norm(left[:, 9], axis=1), 1, atol=1e-4)  # wrist->middle knuckle = 1
    # the left hand only translates in the fixture, so its hand-frame shape must not move at all
    assert np.allclose(left, left[0], atol=1e-4)


def test_missing_hand_is_zero_filled_and_flagged(clip):
    hands, pose = clip
    f = normalize(hands, pose, W)
    mask_l, mask_r, mask_pose = f[:, MASK_SLICE].T
    assert mask_l.all() and mask_pose.all()
    assert 0 < mask_r.sum() < SEQ_LEN                    # right hand present only part of the time
    gone = mask_r == 0
    right_body = f[:, N_POSE_F + 42:N_POSE_F + 84]
    right_local = f[:, LOCAL][:, 63:]
    assert (right_body[gone] == 0).all() and (right_local[gone] == 0).all()
    assert (np.abs(right_local[~gone]).sum(axis=1) > 0).all()


def test_no_pose_and_no_hands_still_gives_valid_features():
    hands = np.full((30, 2, 21, 3), np.nan, np.float32)
    pose = np.full((30, 33, 4), np.nan, np.float32)
    f = normalize(hands, pose, W)
    assert f.shape == (SEQ_LEN, N_FEATURES) and (f == 0).all()


def test_resample_lengths_and_nan_handling():
    x = np.arange(10, dtype=np.float32)[:, None]
    assert np.allclose(resample(x, 19)[:, 0], np.linspace(0, 9, 19))   # stretch = linear interp
    assert len(resample(x, 4)) == 4 and len(resample(x[:1], 64)) == 64
    x[3] = np.nan
    out = resample(x, 19)[:, 0]
    assert np.isnan(out).sum() >= 1                       # the gap survives ...
    assert set(out[np.isfinite(out)] % 0.5) == {0.0}      # ... and never blends into neighbours


def test_mirror_twice_is_identity_and_swaps_hands(clip):
    hands, pose = clip
    h1, p1 = mirror(hands, pose, W)
    assert np.isnan(h1[10:20, 0]).all() and np.isfinite(h1[10:20, 1]).all()   # gap moved slots
    h2, p2 = mirror(h1, p1, W)
    assert np.allclose(h2, hands, equal_nan=True) and np.allclose(p2, pose)


def test_augmented_clip_normalises_cleanly(clip):
    rng = np.random.default_rng(0)
    for _ in range(20):
        f = normalize(*augment(*clip, W, W, rng), W)
        assert f.shape == (SEQ_LEN, N_FEATURES) and np.isfinite(f).all()
