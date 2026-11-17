"""Raw landmarks (pixels, NaN = missing) -> the fixed-size feature array the classifier reads.

In:  hands (T, 2, 21, 3), pose (T, 33, 4) from landmarks.py, plus frame width.
Out: float32 (64, 227), every value in [-FEATURE_CLIP, FEATURE_CLIP], no NaN.

Per frame the 227 numbers are, in order:
  14   body:  nose, shoulders, elbows, wrists (x, y) in the BODY frame
  84   both hands, 21 points (x, y) each, in the BODY frame      -> where the hands are
  126  both hands, 21 points (x, y, z) each, in the HAND frame   -> what shape they make
  3    mask flags: left hand present, right hand present, pose present

BODY frame: origin = midpoint of the shoulders, unit = shoulder width. Removes where the
signer stands and how far they are from the camera.
HAND frame: origin = that hand's wrist, unit = wrist-to-middle-knuckle length (3D). Removes
hand position and size, leaving only handshape and orientation. This is the pose-invariant
3D hand feature the brief asks for.

Missing data is zero-filled and flagged by the mask, so "hand not visible" is a state the
model has seen in training, not garbage coordinates.
"""
import numpy as np

from src.config import FEATURE_CLIP, SEQ_LEN

POSE_KEEP = [0, 11, 12, 13, 14, 15, 16]   # nose, L/R shoulder, L/R elbow, L/R wrist
L_SHOULDER, R_SHOULDER = 11, 12
WRIST, MIDDLE_MCP = 0, 9                  # hand landmark indices
N_POSE_F = len(POSE_KEEP) * 2             # 14
N_HAND_BODY_F = 2 * 21 * 2                # 84
N_HAND_LOCAL_F = 2 * 21 * 3               # 126
N_FEATURES = N_POSE_F + N_HAND_BODY_F + N_HAND_LOCAL_F + 3   # 227
MASK_SLICE = slice(N_FEATURES - 3, N_FEATURES)
EPS = 1e-6


def resample(arr, n=SEQ_LEN):
    """Stretch or squeeze the time axis to exactly n frames by linear interpolation.

    NaN-aware: between a present and a missing frame we take the nearer one instead of
    blending, so a missing hand never leaks half-NaN coordinates into its neighbours.
    """
    arr = np.asarray(arr, np.float32)
    T = len(arr)
    if T == n:
        return arr.copy()
    if T == 1:
        return np.repeat(arr, n, axis=0)
    pos = np.linspace(0, T - 1, n)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, T - 1)
    w = (pos - lo).astype(np.float32).reshape((-1,) + (1,) * (arr.ndim - 1))
    a, b = arr[lo], arr[hi]
    out = a * (1 - w) + b * w
    nearest = np.where(w < 0.5, a, b)
    return np.where(np.isnan(out), nearest, out).astype(np.float32)


def body_frame(pose, width):
    """Return (centre (2,), scale, pose_ok (T,)) for a clip.

    One centre and one scale per clip (median over frames), not per frame: the signer does
    not change position or distance within a two-second sign, and per-frame values would
    inject shoulder jitter into every hand coordinate.
    With no pose in any frame we fall back to the image centre and a quarter of the width.
    """
    ls, rs = pose[:, L_SHOULDER, :2], pose[:, R_SHOULDER, :2]
    ok = np.isfinite(ls).all(1) & np.isfinite(rs).all(1)
    if not ok.any():
        return np.array([width / 2, width / 2], np.float32), float(width) / 4, ok
    centre = np.median((ls[ok] + rs[ok]) / 2, axis=0)
    scale = float(np.median(np.linalg.norm(ls[ok] - rs[ok], axis=1)))
    return centre.astype(np.float32), max(scale, 0.02 * float(width)), ok


def normalize(hands, pose, width, seq_len=SEQ_LEN):
    """Raw clip -> (seq_len, N_FEATURES) float32. See the module docstring for the layout."""
    hands = resample(hands, seq_len)
    pose = resample(pose, seq_len)
    centre, scale, pose_ok = body_frame(pose, float(width))

    body = (pose[:, POSE_KEEP, :2] - centre) / scale                    # (T, 7, 2)
    body[~pose_ok] = 0.0

    hand_ok = np.isfinite(hands).all(axis=(2, 3))                       # (T, 2)
    hands_body = (hands[..., :2] - centre) / scale                      # (T, 2, 21, 2)
    wrist = hands[:, :, WRIST:WRIST + 1, :]                             # (T, 2, 1, 3)
    palm = np.linalg.norm(hands[:, :, MIDDLE_MCP, :] - hands[:, :, WRIST, :], axis=-1)
    hands_local = (hands - wrist) / np.maximum(palm, EPS)[:, :, None, None]
    hands_body[~hand_ok] = 0.0
    hands_local[~hand_ok] = 0.0

    T = seq_len
    feat = np.concatenate([
        body.reshape(T, -1), hands_body.reshape(T, -1), hands_local.reshape(T, -1),
        hand_ok.astype(np.float32), pose_ok.astype(np.float32)[:, None]], axis=1)
    feat = np.nan_to_num(feat, nan=0.0, posinf=FEATURE_CLIP, neginf=-FEATURE_CLIP)
    return np.clip(feat, -FEATURE_CLIP, FEATURE_CLIP).astype(np.float32)
