"""Tests for the parts of landmarks.py that don't need the MediaPipe model files."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.landmarks import assign_hands, read_frames

# Un-mirrored camera: signer's left wrist is on the image's RIGHT (x=300), right wrist at x=100.
POSE_WRISTS = np.array([[300, 200], [100, 200]], np.float32)


def test_two_hands_matched_to_pose_even_if_labels_wrong():
    # labels deliberately both wrong; pose geometry must win
    assert assign_hands([(105, 210), (295, 190)], ["Left", "Right"], POSE_WRISTS) == [1, 0]


def test_one_hand_goes_to_nearest_pose_wrist():
    assert assign_hands([(290, 205)], ["Right"], POSE_WRISTS) == [0]


def test_no_pose_uses_labels():
    assert assign_hands([(10, 10), (20, 20)], ["Right", "Left"], None) == [1, 0]


def test_no_pose_duplicate_labels_falls_back_to_x():
    assert assign_hands([(100, 0), (300, 0)], ["Left", "Left"], None) == [1, 0]


def test_nan_pose_treated_as_missing():
    nan = np.full((2, 2), np.nan, np.float32)
    assert assign_hands([(10, 10)], ["Left"], nan) == [0]


def test_no_hands():
    assert assign_hands([], [], POSE_WRISTS) == []


def _write_video(path, n=40):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (64, 48))
    for i in range(n):
        vw.write(np.full((48, 64, 3), i * 5, np.uint8))  # brightness encodes frame number
    vw.release()


def test_bounds_applied_on_untrimmed_video(tmp_path):
    p = tmp_path / "v.mp4"; _write_video(p)
    frames, idx, fps, mode = read_frames(p, frame_start=11, frame_end=20)
    assert mode == "bounds" and idx[0] == 10 and idx[-1] == 19 and len(frames) == 10


def test_pretrimmed_video_used_whole(tmp_path):
    # json says frames 100..140, but the file only has 40: it was already cut, use all of it
    p = tmp_path / "v.mp4"; _write_video(p)
    _, idx, _, mode = read_frames(p, frame_start=100, frame_end=140)
    assert mode == "whole" and len(idx) == 40


def test_long_clip_subsampled(tmp_path):
    p = tmp_path / "v.mp4"; _write_video(p, n=40)
    frames, idx, _, _ = read_frames(p, max_frames=16)
    assert len(frames) == 16 and idx[0] == 0 and idx[-1] == 39
