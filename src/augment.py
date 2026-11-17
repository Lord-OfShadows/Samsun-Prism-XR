"""Training-time augmentation, applied to RAW landmarks (pixels) before normalize().

Why on raw landmarks and not on video: it is thousands of times cheaper (a clip is a few
kilobytes of numbers), so every epoch sees a freshly perturbed copy of every clip. With
about 14 training clips per word this is the main defence against memorising the data.

Each function takes and returns (hands (T, 2, 21, 3), pose (T, 33, 4)); NaN = missing.
The list is the one in the PRD: rotation, scaling, mirroring, time stretching, random
frame drops, landmark jitter, whole-hand dropout.
"""
import numpy as np

# MediaPipe pose points come in left/right pairs; mirroring must swap them.
_POSE_SWAP = np.arange(33)
for _a, _b in [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18),
               (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]:
    _POSE_SWAP[_a], _POSE_SWAP[_b] = _b, _a


def mirror(hands, pose, width):
    """Flip left-right, as if the signer were left-handed. Hands swap slots too."""
    hands, pose = hands[:, ::-1].copy(), pose[:, _POSE_SWAP].copy()
    hands[..., 0] = width - hands[..., 0]
    pose[..., 0] = width - pose[..., 0]
    return hands, pose


def rotate_scale(hands, pose, width, height, rng, max_deg=15.0, max_scale=0.15):
    """Tilt the whole skeleton about the image centre and stretch x and y a little.

    Rotation imitates a tilted camera or head. The x/y stretch is deliberately uneven:
    uniform scaling would be cancelled exactly by the shoulder-width normalisation.
    """
    th = np.deg2rad(rng.uniform(-max_deg, max_deg))
    sx, sy = rng.uniform(1 - max_scale, 1 + max_scale, size=2)
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], np.float32)
    m = (rot * np.array([sx, sy], np.float32)).T        # row-vector convention: p @ m
    c = np.array([width / 2, height / 2], np.float32)
    hands, pose = hands.copy(), pose.copy()
    hands[..., :2] = (hands[..., :2] - c) @ m + c
    pose[..., :2] = (pose[..., :2] - c) @ m + c
    return hands, pose


def time_warp(hands, pose, rng, max_crop=0.12, strength=0.3):
    """Time stretching. A uniform speed change would be undone by resampling to 64 frames,
    so instead we (a) trim a random bit off each end and (b) play some parts of the sign
    faster and others slower (a smooth, monotonic warp of the time axis)."""
    T = len(hands)
    if T < 8:
        return hands, pose
    lo = rng.uniform(0, max_crop) * (T - 1)
    hi = (T - 1) - rng.uniform(0, max_crop) * (T - 1)
    steps = np.exp(rng.normal(0, strength, size=8))           # 8 segments of random speed
    knots = np.concatenate([[0], np.cumsum(steps)]) / steps.sum()
    pos = lo + (hi - lo) * np.interp(np.linspace(0, 1, T), np.linspace(0, 1, 9), knots)
    idx = np.clip(np.round(pos).astype(int), 0, T - 1)        # nearest frame: never blends NaN
    return hands[idx], pose[idx]


def drop_frames(hands, pose, rng, max_frac=0.2):
    """Delete a random subset of frames, as if the camera had skipped them."""
    T = len(hands)
    n_drop = int(rng.uniform(0, max_frac) * T)
    if n_drop == 0 or T - n_drop < 8:
        return hands, pose
    keep = np.sort(rng.choice(T, size=T - n_drop, replace=False))
    return hands[keep], pose[keep]


def jitter(hands, pose, width, rng, sigma=0.004):
    """Small Gaussian noise on every point (sigma = 0.4% of frame width), imitating
    landmark detector noise."""
    s = sigma * width
    hands = hands + rng.normal(0, s, hands.shape).astype(np.float32)
    pose = pose.copy()
    pose[..., :3] += rng.normal(0, s, pose[..., :3].shape).astype(np.float32)
    return hands, pose


def hand_dropout(hands, rng, p=0.25, min_frac=0.1, max_frac=0.5):
    """Whole-hand dropout: with probability p per hand, make that hand vanish for a
    contiguous stretch of the clip, as if it were occluded or the detector lost it.
    Never the entire clip, so the label stays recoverable."""
    hands = hands.copy()
    T = len(hands)
    for slot in (0, 1):
        if rng.random() < p:
            n = max(1, int(rng.uniform(min_frac, max_frac) * T))
            start = rng.integers(0, T - n + 1)
            hands[start:start + n, slot] = np.nan
    return hands


def augment(hands, pose, width, height, rng):
    """The full training augmentation for one clip."""
    hands, pose = time_warp(hands, pose, rng)
    hands, pose = drop_frames(hands, pose, rng)
    if rng.random() < 0.5:
        hands, pose = mirror(hands, pose, width)
    hands, pose = rotate_scale(hands, pose, width, height, rng)
    hands, pose = jitter(hands, pose, width, rng)
    hands = hand_dropout(hands, rng)
    return hands, pose
