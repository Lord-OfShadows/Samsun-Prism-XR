# Status: Core tier build

Written 17 Nov 2026, about 13:10 IST, when the build session was paused. Every number
here was printed by a script run in that session; the script and output file are named
next to each one. The README's Results section is regenerated from the same files by
`python scripts/update_readme.py`.

## Headline

- The whole Core pipeline exists as working code and was run on the **real WLASL100 data**
  (all 2,038 clips, official split).
- **INT8 model, WLASL100 test split (258 clips): top-1 82.56 %, top-5 93.41 %.**
  Float model: top-1 83.33 %, top-5 93.80 %. Quantization cost 0.78 points of top-1.
  Source: `python -m src.convert` -> `outputs/quantization.json`.
- INT8 classifier file: **1.82 MB**, fully integer (0 float tensors left).
- The same test clips through `predict()` on the video files: **top-1 82.56 %, top-5
  93.41 %**, 258 of 258 clips answered, 0 errors. Source: `scripts/make_predictions.py` ->
  `outputs/predictions.jsonl`, scored by `scripts/make_results.py`.
- **Still running unattended when this was written:** the latency benchmark and the
  robustness evaluation, after which `results.csv` is rewritten. See "What is left".

## What works (run and verified)

| Piece | Evidence |
|---|---|
| Environment | `requirements.txt` installed into a fresh Python 3.11.18 venv; MediaPipe 1.2.3 and TensorFlow 2.21.0 coexist (numpy 2.4.6, protobuf 7.36.2). |
| Dataset download | `scripts/get_wlasl100.py`: 2,038 clips, none missing against `WLASL_v0.3.json`. |
| Landmark extraction | `scripts/extract_all.py` over all 2,038 clips: 2,036 usable, 2 with no hand found (1 train, 1 val, 0 test). Report: `outputs/extraction_report.md`. |
| Normalisation, augmentation | Unit tests in `tests/test_normalize.py` (position and distance invariance, masking, mirroring). |
| Model and training | `python -m src.train`: 1,586,596 parameters, 250 epochs, best validation epoch 190. Log: `outputs/train_log.json`. |
| INT8 conversion | `python -m src.convert`: attention layers converted cleanly; no fallback architecture needed, no quantization-aware training needed (drop under 2 points). |
| `predict()` | Returned valid JSON for all 258 test clips, offline, 0 errors (`outputs/predictions.jsonl`). |
| Results table | `scripts/make_results.py` wrote `outputs/results.csv`; the accuracy row is filled, the latency, size and robustness rows say "not measured" until their runs finish. |
| Benchmark, robustness scripts | Each run end to end on a trial model with a few clips, outputs sent to a temp folder. Full runs in progress. |
| Live demo loop | `python demo/live.py --source <clip> --no-window` processes a clip and prints captions. Not yet seen by a human at a webcam. |
| Tests | `python -m pytest -q`: 30 passed on the last full run. After the final small refactors the 24 non-INT8 tests were rerun and passed; the 6 INT8 tests should be rerun once the machine is idle. |

## Measured numbers

### Data (`outputs/extraction_report.md`, `outputs/clip_counts.csv`)

| | Train | Val | Test |
|---|---|---|---|
| Clips in the official split | 1,442 | 338 | 258 |
| Usable (a hand found in at least one frame) | 1,441 | 337 | 258 |
| Usable clips per word: min / median / max | 12 / 14 / 30 | 2 / 3 / 6 | 2 / 3 / 5 |

- Hands were found in a median of 67 % of a clip's frames (mean 69 %); the body in 100 %.
- 384 clips have a hand in fewer than half their frames. Many clips begin and end with the hands at rest, so this is expected, but it was not investigated further.

### Accuracy (`outputs/quantization.json`)

Every clip in the split counts, including the one validation clip with no hand found.

| Split | Clips | Float top-1 | INT8 top-1 | Float top-5 | INT8 top-5 | Same top word |
|---|---|---|---|---|---|---|
| val | 338 | 85.21 % | 85.21 % | 94.97 % | 94.38 % | 99.11 % |
| test | 258 | 83.33 % | 82.56 % | 93.80 % | 93.41 % | 98.84 % |

These are measured on the cached landmarks. The figure in `results.csv` is measured a
second way, by calling `predict()` on each test video file, and came out the same:
82.56 % top-1 and 93.41 % top-5 on the test split.

One training run, one seed. No hyperparameter tuning was done, so the test number is not
tuned to the test set, but its run-to-run spread is unknown.

### Size

| File | Bytes | MB |
|---|---|---|
| `models/echogesture_int8.tflite` | 1,822,904 | 1.82 |
| `models/mediapipe/hand_landmarker.task` | 7,819,105 | 7.82 |
| `models/mediapipe/pose_landmarker_lite.task` | 5,777,746 | 5.78 |
| Sum | 15,419,755 | 15.42 |

Under the 20 MB budget with the pose model included, so the hands-only fallback is not
needed. `src/benchmark.py` computes this sum itself and writes it to `outputs/benchmark.json`.

### Latency, robustness

**Not available when this was written**: the runs were in progress. See below.

### Not measured, and will not be in Core

`avg_power` and `thermal_drift_10min` need an on-phone run. There is no Android build.
They are written as "not measured" in `results.csv`.

## What is left

### 1. Collect the unattended run (first thing to do)

A pipeline was left running. Predictions are finished and committed; still to come are
benchmark -> robustness -> results -> readme.
Its log is `logs/pipeline.log`. When it has finished (the last line says `readme done`):

```
git add outputs README.md
git commit -m "Add benchmark, robustness and final results"
git push origin core-mvp
```

If it stopped part-way, the log's last line names the failed step; resume with
`python scripts/run_all.py --from <step>`.

Then update this file's "Latency, robustness" section from the README Results block.

### 2. Expect the latency numbers from this laptop to look bad

The laptop throttles hard under sustained load. In this session MediaPipe measured about
50 ms per frame when cool and 170 to 230 ms per frame once throttled, about 15 seconds
into a run. The 200-run benchmark will mostly sit in the throttled state. Rerun
`python -m src.benchmark` on a cool, idle machine before quoting latency anywhere
(`docs/BLOCKED.md` #4). Whatever the machine, a whole clip is 64 frames of two MediaPipe
models and will not be under 50 ms; only the classifier call (single-digit milliseconds
in trial runs) is. Which of these the brief's 50 ms means is an open question to the
organisers.

### 3. Human checks

- Sign at the webcam: `python demo/live.py`. Tune the threshold and hold time by eye.
- Open a few previews to confirm left and right hands are assigned correctly:
  `python scripts/extract_all.py --limit 12 --preview --overwrite` writes pictures to `data/preview/`.
- Confirm the items marked **CHECK** in `docs/ASSUMPTIONS.md`, above all the output formats (#2) and the latency definition (#20).
- Read `docs/MODULES.md` and make sure each owner can explain their modules.

### 4. Small loose ends

- `models/echogesture_float.keras` is 19.3 MB because it includes the optimizer state. Saving it without the optimizer would bring it to about 6 MB. It is only needed to rerun `src/convert.py`; the shipped model is the 1.82 MB INT8 file.
- The landmark zip for teammates (`data/share/echogesture_wlasl100_landmarks.zip`, 80.7 MB) exists only on the build machine. Send it to the team directly.
- The PRD PDF in `docs/` is not committed (public repo).

## PRD Core checklist

| Core requirement | State |
|---|---|
| `predict()` returns valid JSON for any clip in the test manifest, with no network access | **Done.** All 258 test clips answered, 0 errors, models loaded from local files only. |
| INT8 `.tflite` model and `convert.py` in the repo | **Done.** |
| `make_predictions.py` writes `predictions.jsonl` with every clip ID present | **Done.** 258 lines for 258 clips. A clip that failed would still get a line. |
| `make_results.py` writes `results.csv` and computes the pass column | **Done for accuracy.** The latency, size and robustness rows are filled when the unattended run ends. |
| Measured top-1 and top-5 on the WLASL100 test split | **Done:** INT8 82.56 % / 93.41 %, both on cached landmarks and through `predict()`. |
| Measured p95 latency, split into landmark extraction and classification | **Script done and trial-run**; 200-run measurement in progress. Rerun on a cool machine. |
| Webcam demo with caption overlay, and the recorded demo video | **Code done**, tested headless on video files. A human still has to try the webcam. The recorded video is out of scope for this build. |
| README with setup, commands and the stated vocabulary assumption | **Done**, with diagrams and an AI-assistance line. |
| PPT on the organisers' template | Out of scope for this build. |

Also built, beyond the Core list because the brief asked for it: the robustness evaluation
(`src/robustness.py`), which the PRD places in the Target tier.

## What the team must run on real data

The real-data runs are done or in progress on the build machine, so nothing has to be
rerun just to get numbers. Each owner should still run their part once, to be able to
defend it:

| Owner | Run | Look at |
|---|---|---|
| A: data and landmarks | `python scripts/get_wlasl100.py`, then `python scripts/extract_all.py --limit 12 --preview --overwrite` | The preview pictures in `data/preview/`; `outputs/extraction_report.md` |
| B: model and training | Unpack the landmark zip, `python -m src.train`, `python -m src.convert` | Whether a second run lands near 83 %; `outputs/train_log.json` |
| C: quantization, predict, benchmark, demo | `python predict.py <clip>`, `python -m src.benchmark` on a cool machine, `python demo/live.py` | Per-stage latency table; caption stability at the webcam |
