"""Video -> raw keypoint arrays, using the MediaPipe Tasks API (HandLandmarker + PoseLandmarker).

In:  a video path, optionally WLASL frame bounds.
Out: dict with
     hands     float32 (T, 2, 21, 3)  slot 0 = signer's LEFT hand, slot 1 = signer's RIGHT hand
     pose      float32 (T, 33, 4)     x, y, z, visibility for all 33 pose points
     frame_idx int32   (T,)           which original frames (0-based) were processed
     fps, width, height, bounds_mode
Coordinates are in pixels of the processed frame (x*W, y*H, z*W) so x and y share one
scale even on non-square video. Missing hands / no person = NaN.

Deliberately NOT done here: centring, scaling, zero-filling, resampling to 64 frames.
That is normalize.py. Extraction is the slow step; keeping it "raw" means we can change
the normalisation later without re-running MediaPipe over 2,000 videos.
"""
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

N_HAND_PTS, N_POSE_PTS = 21, 33
POSE_L_WRIST, POSE_R_WRIST = 15, 16   # MediaPipe pose indices; "left" = the signer's left

MAX_FRAMES = 192   # cap per clip; WLASL signs are a few seconds, this only bites on long clips
MAX_SIDE = 640     # downscale big frames first: MediaPipe resizes internally anyway, and
                   # converting a 1280x720 frame costs time for no landmark gain
# 0.5 is MediaPipe's default. WLASL is low-res and blurry, so we may lower these if the
# five-clip test shows many frames with no hand found. Decide from the numbers, not now.
HAND_DET_CONF = 0.5
POSE_DET_CONF = 0.5


def read_frames(path, frame_start=None, frame_end=None, max_frames=MAX_FRAMES):
    """Return (frames_bgr list, frame_idx array, fps, bounds_mode).

    WLASL bounds are 1-based and inclusive; frame_end == -1 means "to the end".
    Preprocessed WLASL mirrors already cut each clip to its bounds, while raw downloads do
    not. So bounds are applied only if the video is long enough to contain frame_end;
    otherwise the clip is assumed pre-trimmed and used whole. bounds_mode records which.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 1 else 25.0          # some files report 0 fps
    n_reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    lo, hi, mode = 0, None, "whole"
    if frame_end is not None and frame_end > 0 and n_reported >= frame_end:
        lo, hi, mode = max(int(frame_start or 1) - 1, 0), int(frame_end) - 1, "bounds"
    elif frame_end == -1 and frame_start and frame_start > 1 and n_reported >= frame_start:
        lo, mode = int(frame_start) - 1, "bounds"

    frames, idx, i = [], [], 0
    while True:
        ok, frame = cap.read()
        if not ok or (hi is not None and i > hi):
            break
        if i >= lo:
            frames.append(frame)
            idx.append(i)
        i += 1
    cap.release()

    if len(frames) > max_frames:   # uniform subsample keeps the whole sign, just sparser
        keep = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
        frames = [frames[k] for k in keep]
        idx = [idx[k] for k in keep]
    return frames, np.asarray(idx, np.int32), fps, mode


def assign_hands(wrists, labels, pose_wrists):
    """Decide which detected hand is the signer's left (slot 0) and right (slot 1).

    wrists:      list of (x, y) wrist positions of the detected hands (0, 1 or 2 of them)
    labels:      MediaPipe handedness label per hand, "Left" / "Right"
    pose_wrists: array (2, 2) = pose [left wrist, right wrist] in the same pixels, or None

    Why not just trust the handedness label: it is a per-frame guess from the hand crop
    alone and flips when hands cross or blur, which is constant in signing. The pose model
    sees the whole arm, so matching each hand to the nearest pose wrist is more stable.
    The label is only the fallback when there is no pose.
    Returns a list of slots, one per detected hand.
    """
    n = len(wrists)
    if n == 0:
        return []
    w = np.asarray(wrists, np.float32)[:2]
    if pose_wrists is not None and np.all(np.isfinite(pose_wrists)):
        pw = np.asarray(pose_wrists, np.float32)
        d = np.linalg.norm(w[:, None, :] - pw[None, :, :], axis=-1)   # (n, 2)
        if n == 1:
            return [int(np.argmin(d[0]))]
        # two hands: pick the pairing with the smaller total distance
        return [0, 1] if d[0, 0] + d[1, 1] <= d[0, 1] + d[1, 0] else [1, 0]

    slots = [0 if lab == "Left" else 1 for lab in labels[:2]]
    if n >= 2 and slots[0] == slots[1]:
        # both labelled the same side: fall back to image position. In un-mirrored video
        # the signer's left hand appears on the image's right (larger x).
        return [0, 1] if w[0, 0] > w[1, 0] else [1, 0]
    return slots


class LandmarkExtractor:
    """Runs both landmarkers over one clip. Create once per worker process, reuse per clip."""

    def __init__(self, hand_model, pose_model):
        for p in (hand_model, pose_model):
            if not Path(p).exists():
                raise FileNotFoundError(f"{p} missing; run scripts/get_mediapipe_models.py")
        self.hand_model, self.pose_model = str(hand_model), str(pose_model)

    def _landmarkers(self):
        # VIDEO mode tracks hands between frames instead of re-detecting every frame,
        # which is faster and less jittery. Its tracking state must not leak from one clip
        # into the next, so we build fresh landmarkers per clip (small fixed cost).
        hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.hand_model),
            running_mode=vision.RunningMode.VIDEO, num_hands=2,
            min_hand_detection_confidence=HAND_DET_CONF))
        pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.pose_model),
            running_mode=vision.RunningMode.VIDEO, num_poses=1,
            min_pose_detection_confidence=POSE_DET_CONF))
        return hand, pose

    def extract_frames(self, frames_bgr, fps):
        T = len(frames_bgr)
        hands = np.full((T, 2, N_HAND_PTS, 3), np.nan, np.float32)
        pose = np.full((T, N_POSE_PTS, 4), np.nan, np.float32)
        hand_lm, pose_lm = self._landmarkers()
        with hand_lm, pose_lm:
            last_ts = -1
            for t, frame in enumerate(frames_bgr):
                h, w = frame.shape[:2]
                rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts = max(int(t * 1000 / fps), last_ts + 1)   # VIDEO mode needs strictly increasing ms
                last_ts = ts

                pr = pose_lm.detect_for_video(image, ts)
                if pr.pose_landmarks:
                    pose[t] = [[p.x * w, p.y * h, p.z * w, p.visibility or 0.0]
                               for p in pr.pose_landmarks[0]]

                hr = hand_lm.detect_for_video(image, ts)
                pts = [np.array([[p.x * w, p.y * h, p.z * w] for p in lms], np.float32)
                       for lms in hr.hand_landmarks]
                labels = [c[0].category_name for c in hr.handedness]
                pose_wrists = pose[t, [POSE_L_WRIST, POSE_R_WRIST], :2]
                for k, slot in enumerate(assign_hands([p[0, :2] for p in pts], labels, pose_wrists)):
                    hands[t, slot] = pts[k]
        return hands, pose

    def extract(self, video_path, frame_start=None, frame_end=None):
        frames, idx, fps, mode = read_frames(video_path, frame_start, frame_end)
        if not frames:
            raise IOError(f"no frames decoded from {video_path}")
        h, w = frames[0].shape[:2]
        if max(h, w) > MAX_SIDE:
            s = MAX_SIDE / max(h, w)
            frames = [cv2.resize(f, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
                      for f in frames]
            h, w = frames[0].shape[:2]
        hands, pose = self.extract_frames(frames, fps)
        return {"hands": hands, "pose": pose, "frame_idx": idx, "fps": np.float32(fps),
                "width": np.int32(w), "height": np.int32(h), "bounds_mode": mode,
                "_preview_frame": frames[len(frames) // 2]}


def draw_preview(frame, hands_t, pose_t):
    """One frame with points drawn, to check by eye that left/right is assigned correctly.
    Green = slot 0 (signer's left), red = slot 1 (signer's right), blue = pose."""
    img = frame.copy()
    for (x, y, *_ ) in pose_t[:25]:
        if np.isfinite(x):
            cv2.circle(img, (int(x), int(y)), 3, (255, 0, 0), -1)
    for slot, color in ((0, (0, 200, 0)), (1, (0, 0, 255))):
        for (x, y, _) in hands_t[slot]:
            if np.isfinite(x):
                cv2.circle(img, (int(x), int(y)), 2, color, -1)
    cv2.putText(img, "green=L red=R", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return img
