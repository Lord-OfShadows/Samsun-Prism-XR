"""Shared synthetic fixtures. No dataset, no MediaPipe model files and no network needed."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

W = H = 256.0


def make_clip(T=40, seed=0, shift=(0.0, 0.0), zoom=1.0):
    """A plausible raw clip in pixels: a body, two moving hands, right hand missing for a while.

    shift / zoom move and resize the WHOLE skeleton, as if the signer stood elsewhere or
    nearer the camera. Normalised features must not change under either.
    """
    rng = np.random.default_rng(seed)
    pose = np.zeros((T, 33, 4), np.float32)
    pose[..., :2] = rng.uniform(60, 200, (1, 33, 2))            # a fixed body
    pose[:, 11, :2], pose[:, 12, :2] = (160, 100), (96, 100)    # L / R shoulder: width 64 px
    pose[..., 3] = 1.0
    t = np.linspace(0, 1, T)[:, None, None]
    shape = rng.normal(0, 12, (1, 2, 21, 3)).astype(np.float32)  # handshape around the wrist
    shape[:, :, 0] = 0
    wrist = np.zeros((T, 2, 1, 3), np.float32)
    wrist[:, 0, 0, :2] = np.array([170, 180]) + 30 * np.concatenate([t[:, 0], -t[:, 0]], 1)
    wrist[:, 1, 0, :2] = np.array([90, 180]) + 20 * np.concatenate([-t[:, 0], -t[:, 0]], 1)
    hands = (wrist + shape).astype(np.float32)
    hands[10:20, 1] = np.nan                                     # right hand lost for 10 frames

    centre = np.array([W / 2, H / 2], np.float32)
    for arr in (hands, pose):
        arr[..., :2] = (arr[..., :2] - centre) * zoom + centre + np.asarray(shift, np.float32)
    hands[..., 2] *= zoom                                        # z scales with the hand too
    return hands, pose


@pytest.fixture
def clip():
    return make_clip()
