# EchoGesture

On-device isolated sign recognition: one short video clip of one sign in, one English word
out. Entry for Track 2 of the Samsung PRISM XR Hackathon 2026.

```
video -> MediaPipe hand + pose landmarks -> normalise -> INT8 classifier -> JSON
```

The video is reduced to keypoints first and only the keypoints are classified. No image
is stored and nothing leaves the device.

**What this is not:** it does not translate continuous signing, and it only knows the 100
words in `configs/vocab.txt`.

## Vocabulary assumption

The organisers have not supplied a word list, so we assume **WLASL100**: the first 100
glosses of `WLASL_v0.3.json`, with the **official train / val / test split** stored in
that file (1,442 / 338 / 258 clips). The model is trained on train, selected on val and
reported on test. The word list lives in one file, `configs/vocab.txt`; retraining on a
different list is one command (see "Retrain on another word list").

## Results

Every number below is produced by a script in this repo. Nothing is typed by hand, and
anything we could not measure says so.

<!-- RESULTS:START -->
Results have not been generated yet. See `docs/STATUS.md`.
<!-- RESULTS:END -->

## Setup

Python **3.11** is required (Keras 3.15 does not support 3.10). Tested on Windows 11; CPU
only, no GPU needed.

```
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python scripts/get_mediapipe_models.py
```

The last line downloads the two MediaPipe model files (13.6 MB) into `models/mediapipe/`.
It is the only step that needs the network; everything after it runs offline.

## Recognise a clip

```
python predict.py path/to/clip.mp4
```

or from Python:

```python
from predict import predict
result = predict("path/to/clip.mp4")
```

Output (the schema is defined in one function, `format_output()` in `predict.py`):

```json
{
  "clip_id": "clip",
  "prediction": "book",
  "confidence": 0.83,
  "top5": [{"word": "book", "confidence": 0.83}, {"word": "...", "confidence": 0.05}],
  "latency_ms": {"decode": 0.0, "landmarks": 0.0, "normalize": 0.0, "classifier": 0.0, "total": 0.0},
  "frames": 64
}
```

The values above only illustrate the format.

## Live webcam demo

```
python demo/live.py                    # press q to quit
python demo/live.py --source clip.mp4  # run a video file through the same loop
```

A word appears only when the model's confidence is above a threshold (default 0.6) and
then stays on screen for at least 1.5 seconds, so the caption does not flicker. Both
values are in `src/config.py`.

## Reproduce every number

### 1. Get the data

You need `WLASL_v0.3.json` and a folder of `<video_id>.mp4` files for the 100 words.

- Metadata: `https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json`
- Videos: we used the public Hugging Face mirror `aipieces/WLASL100` (five zip files under
  `WLASL100/`, about 212 MB, 2,038 clips already trimmed to 256x256). Unzip them and put
  all the `.mp4` files in one flat folder. Check the dataset's licence before
  redistributing anything.

Place them at `data/wlasl/WLASL_v0.3.json` and `data/wlasl/videos/`. `data/` is not
committed.

### 2. Run the pipeline

Everything, in one command:

```
python scripts/run_all.py --wlasl-json data/wlasl/WLASL_v0.3.json --videos data/wlasl/videos
```

Or step by step:

| Step | Command | Writes |
|---|---|---|
| Cache landmarks | `python scripts/extract_all.py --wlasl-json data/wlasl/WLASL_v0.3.json --videos data/wlasl/videos` | `data/landmarks/`, `data/manifest.csv`, `outputs/extraction_report.md`, `outputs/clip_counts.csv` |
| Train | `python -m src.train` | `models/echogesture_float.keras`, `outputs/train_log.json` |
| Convert to INT8 | `python -m src.convert` | `models/echogesture_int8.tflite`, `outputs/quantization.json` |
| Predictions | `python scripts/make_predictions.py` | `outputs/predictions.jsonl` |
| Latency and size | `python -m src.benchmark` | `outputs/benchmark.json` |
| Robustness | `python -m src.robustness` | `outputs/robustness.json` |
| Results table | `python scripts/make_results.py` | `outputs/results.csv` |
| Tests | `python -m pytest -q` | |

Landmark extraction is the slow step (about an hour for the 2,038 clips on our laptop);
it is resumable. To write a fresh `configs/vocab.txt` from the first N glosses of the
json: `python scripts/extract_all.py --wlasl-json ... --videos ... --write-vocab 100`.

### Retrain on another word list

Edit `configs/vocab.txt` (one word per line; each must be a gloss in the WLASL json), then
run the `run_all.py` command above. The number of classes is read from that file.

## Repository layout

```
predict.py              the function the evaluator runs
configs/vocab.txt       the word list (the only place it is defined)
src/
  config.py             paths and constants
  landmarks.py          video -> keypoints (MediaPipe)
  normalize.py          centring, scaling, masking -> 64 x 227 features
  augment.py            training-time augmentation
  model.py              network definition
  train.py              training loop
  convert.py            Keras -> full-integer INT8 TFLite
  classifier.py         runs the INT8 model (input scaling lives here)
  benchmark.py          latency and size
  robustness.py         perturbed test set and accuracy drop
  data.py               manifest and cache readers
scripts/
  get_mediapipe_models.py, extract_all.py, make_predictions.py, make_results.py, run_all.py
demo/live.py            webcam captions
models/                 echogesture_int8.tflite (+ the float model it was converted from)
outputs/                predictions.jsonl, results.csv and the json files behind them
docs/                   STATUS.md, MODULES.md, ASSUMPTIONS.md
tests/
```

Plain-language notes on every module are in `docs/MODULES.md`. Decisions made where the
brief was unclear are in `docs/ASSUMPTIONS.md`. Current state and open items are in
`docs/STATUS.md`.

## AI assistance

An AI coding assistant was used for code generation in this project, under the team's
direction. Each module has an owner on the team who ran it, tested it and can explain it.
