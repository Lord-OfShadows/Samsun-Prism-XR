# Blocked and unverified items

What could not be done or checked during the Core build on 17 Nov 2026, and exactly what
a team member needs to run to close each one.

## Not blocked: the dataset

The brief expected the data might be missing. It was missing from the repo, but the full
WLASL100 set was downloaded from public sources within the time limit, so every accuracy
number is from real data. To get the same data on another machine:

```
python scripts/get_wlasl100.py      # WLASL_v0.3.json + 2,038 clips into data/wlasl/
python scripts/extract_all.py       # MediaPipe landmarks into data/landmarks/ (about an hour)
```

or unpack the landmark zip a teammate already built:

```
python scripts/pack_landmarks.py --unpack path\to\echogesture_wlasl100_landmarks.zip
```

## Blocked or unverified

| # | Item | Why | What to run or do |
|---|---|---|---|
| 1 | **Webcam demo, seen by a human.** | The loop was run on video files in headless mode only, and with a trial model, not the trained one; nobody was at the machine to look at the window or sign at the camera. | `python demo/live.py` then sign a few words from `configs/vocab.txt`. Tune `CAPTION_THRESHOLD` and `CAPTION_HOLD_S` in `src/config.py` by eye. |
| 2 | **`avg_power` and `thermal_drift_10min`.** | They need an on-phone run. No Android build exists (stretch goal). | Nothing in Core. Reported as "not measured" in `results.csv`; do not fill with estimates. |
| 3 | **Organisers' exact JSON and CSV formats.** | `docs/BRIEF_FORMATS.md` was not in the repo. | Compare the brief with the schema in `docs/ASSUMPTIONS.md`. If they differ, edit `format_output()` in `predict.py` and `FIELDS` / the row builders in `scripts/make_results.py`, then rerun `python scripts/make_predictions.py` and `python scripts/make_results.py`. |
| 4 | **Latency on a representative device.** | Measured on a laptop CPU that throttles hard under sustained load (MediaPipe went from about 50 ms to about 200 ms per frame after roughly 15 seconds of load during this session). Latency figures from this machine are therefore pessimistic and noisy, and are not phone numbers. | Rerun `python -m src.benchmark` on a cool, idle machine plugged into power, with nothing else running. Then `python scripts/make_results.py` and `python scripts/update_readme.py`. |
| 5 | **Which latency the 50 ms ceiling means.** | Open question to the organisers. | `results.csv` reports the whole-clip figure by default. Change `LATENCY_METRIC_FOR_RESULTS` in `src/config.py` if they confirm per-frame or per-classifier-call, then rerun `python scripts/make_results.py`. |
| 6 | **GPU training.** | TensorFlow has no GPU support on native Windows and MediaPipe's Python package has none on Windows either. Not needed: training takes well under an hour on CPU. | Only if wanted: train inside WSL2 or on Colab with the landmark zip. |
| 7 | **Sharing the landmark zip.** | 80.7 MB, too large for git, and WLASL's licence is for computational use only, so it was not uploaded anywhere public. | The file is at `data/share/echogesture_wlasl100_landmarks.zip` on the machine that built it. Send it to teammates directly (Drive, USB). |
| 8 | **PRD in the repo.** | The PRD exists only as a PDF in `docs/` and was left uncommitted because the repository is public. | Commit it if the team is happy for it to be public. |

## Out of scope for this build (not blocked, just not done)

Target and Stretch tier items, the PPT, the recorded demo video and the Android app.
