"""Live captions from a webcam (or a video file standing in for one).

  python demo/live.py                       # default webcam, press q to quit
  python demo/live.py --source clip.mp4     # play a file through the same loop
  python demo/live.py --source clip.mp4 --no-window   # headless, prints captions (for testing)
  python demo/live.py --no-speak            # captions on screen only, no voice
  python demo/live.py --record demo.mp4     # also save the captioned video (picture only, no sound)

How it works: the newest ~2.5 seconds of landmarks are kept in a sliding window. A few
times per second the window is normalised exactly like a training clip and classified by
the INT8 model. CaptionStabilizer then decides what the viewer sees: a word appears only
when its confidence clears a threshold, and once shown it stays for a minimum time, so
the caption does not flicker between guesses. If no hand has been visible in the window,
nothing is classified at all.

Each new caption is also spoken aloud, offline, by demo/speech.py (Windows' built-in voice,
in a separate process so the video never waits for the audio). Speech is off with
--no-speak, and in --no-window runs, which are for testing.
"""
import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

WINDOW_S = 2.5          # WLASL clips are about 2.5 s (median 61 frames at 25 fps)
CLASSIFY_EVERY_S = 0.3  # how often the window is re-classified
MIN_HAND_FRAC = 0.3     # need a hand in at least this share of the window to classify


class CaptionStabilizer:
    """Turns a noisy stream of (word, confidence) guesses into a steady caption.

    Rules: ignore guesses below `threshold`; once a word is shown, keep it for at least
    `hold_s` seconds before a different word may replace it; clear the caption when
    nothing confident has been seen for `hold_s` seconds.
    """

    def __init__(self, threshold=config.CAPTION_THRESHOLD, hold_s=config.CAPTION_HOLD_S):
        self.threshold, self.hold_s = threshold, hold_s
        self.word, self.confidence = None, 0.0
        self._shown_at = self._last_confident = -1e9

    def update(self, word, confidence, now):
        """Feed one guess (word may be None). Returns the caption to display, or None."""
        confident = word is not None and confidence >= self.threshold
        if confident:
            self._last_confident = now
            if word == self.word:
                self.confidence = confidence
            elif self.word is None or now - self._shown_at >= self.hold_s:
                self.word, self.confidence, self._shown_at = word, confidence, now
        elif self.word is not None and now - max(self._last_confident, self._shown_at) >= self.hold_s:
            self.word, self.confidence = None, 0.0
        return self.word


def draw_caption(frame, word, confidence, fps):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 70), (w, h), (0, 0, 0), -1)
    if word:
        cv2.putText(frame, word.upper(), (20, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
        cv2.putText(frame, f"{confidence:.0%}", (w - 110, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 255, 180), 2)
    cv2.putText(frame, f"{fps:.0f} fps  on-device", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    return frame


class RealTimeWriter:
    """Saves the captioned frames to an .mp4 that plays back at real speed.

    The live loop runs slower than the camera (MediaPipe takes most of each frame), so
    frames arrive unevenly. Each frame is written as many times as needed to fill the
    output slots up to its timestamp, so one second of signing is one second of video.
    For a file source the timestamps are the file's own, so every frame is written once.
    """

    def __init__(self, path, fps):
        self.path, self.fps, self.out, self.n = str(path), fps, None, 0

    def write(self, frame, now):
        if self.out is None:
            h, w = frame.shape[:2]
            self.out = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
            if not self.out.isOpened():
                sys.exit(f"cannot write video to {self.path!r}")
        target = int(now * self.fps) + 1
        while self.n < target:
            self.out.write(frame)
            self.n += 1

    def close(self):
        if self.out is not None:
            self.out.release()
            print(f"saved {self.n / self.fps:.1f} s of video to {self.path} (no sound: the voice is not recorded)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="webcam index or a video file")
    ap.add_argument("--threshold", type=float, default=config.CAPTION_THRESHOLD)
    ap.add_argument("--hold", type=float, default=config.CAPTION_HOLD_S)
    ap.add_argument("--no-window", action="store_true", help="no display; print caption changes")
    ap.add_argument("--no-speak", action="store_true", help="do not speak the captions")
    ap.add_argument("--mirror", action="store_true", help="flip the DISPLAY like a mirror (not the model input)")
    ap.add_argument("--record", metavar="OUT.mp4", help="also save the captioned video to this file")
    args = ap.parse_args()

    from src.classifier import Int8Classifier
    from src.landmarks import LandmarkExtractor, shrink
    from src.normalize import normalize
    from speech import CaptionSpeaker

    is_file = not args.source.isdigit()
    cap = cv2.VideoCapture(args.source if is_file else int(args.source))
    if not cap.isOpened():
        sys.exit(f"cannot open video source {args.source!r}")
    file_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    writer = RealTimeWriter(args.record, file_fps if is_file else 30.0) if args.record else None

    vocab = config.load_vocab()
    clf = Int8Classifier(config.INT8_MODEL)
    extractor = LandmarkExtractor(config.HAND_MODEL, config.POSE_MODEL)
    caption = CaptionStabilizer(args.threshold, args.hold)
    speaker = CaptionSpeaker() if config.SPEAK_CAPTIONS and not (args.no_speak or args.no_window) else None
    if speaker is not None and not speaker.on:
        print(f"speech off: {speaker.reason_off}")
    window = deque()                       # (time, hands, pose)
    last_classified, shown, n, fps = -1e9, None, 0, 0.0
    t_start = t_prev = time.perf_counter()

    with extractor.stream() as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = shrink([frame])[0]
            # A file is "played" at its own frame rate; a webcam uses the wall clock.
            now = n / file_fps if is_file else time.perf_counter() - t_start
            n += 1
            hands, pose = landmarker.process(frame, now * 1000)
            window.append((now, hands, pose))
            while window and now - window[0][0] > WINDOW_S:
                window.popleft()

            if now - last_classified >= CLASSIFY_EVERY_S and len(window) >= 8:
                last_classified = now
                h_arr = np.stack([w[1] for w in window])
                p_arr = np.stack([w[2] for w in window])
                has_hand = np.isfinite(h_arr[:, :, 0, 0]).any(axis=1).mean()
                word, conf = None, 0.0
                if has_hand >= MIN_HAND_FRAC:
                    probs = clf.probs(normalize(h_arr, p_arr, frame.shape[1]))
                    word, conf = vocab[int(probs.argmax())], float(probs.max())
                current = caption.update(word, conf, now)
                if current != shown:
                    shown = current
                    if shown and speaker is not None:
                        speaker.say(shown)
                    print(f"[{now:6.2f}s] caption: {shown or '(none)'}"
                          + (f"  ({caption.confidence:.2f})" if shown else ""), flush=True)

            t = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(t - t_prev, 1e-6) if fps else 1 / max(t - t_prev, 1e-6)
            t_prev = t
            if writer is not None or not args.no_window:
                view = cv2.flip(frame, 1) if args.mirror else frame
                view = draw_caption(view, caption.word, caption.confidence, fps)
                if writer is not None:
                    writer.write(view, now)
                if not args.no_window:
                    cv2.imshow("EchoGesture (q to quit)", view)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    cap.release()
    if writer is not None:
        writer.close()
    if speaker is not None:
        speaker.close()
    if not args.no_window:
        cv2.destroyAllWindows()
    print(f"processed {n} frames in {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    main()
