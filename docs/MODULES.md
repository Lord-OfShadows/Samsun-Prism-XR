# Module notes

One note per module: what it does and why it is built that way, in plain language. The
owner of each module should be able to say this to a jury without reading from the page.

The pipeline in one line:

```
video -> landmarks.py -> normalize.py -> classifier.py (INT8 model) -> predict.py (JSON)
```

Training adds `augment.py`, `model.py`, `train.py` and `convert.py` around that line.

---

## Person A: data and landmarks

### `src/landmarks.py` — video to keypoints

**What it does.** Opens a video with OpenCV and runs two MediaPipe models on each frame: a
hand landmarker (21 3D points per hand) and a pose landmarker (33 body points). It returns
the raw points in pixels, with "not found" stored as NaN.

**Why it is built this way.**
- *Keypoints instead of pixels.* The classifier never sees an image. That makes it tiny, makes it ignore background, clothing and lighting, and means no image is ever stored.
- *At most 64 frames per clip.* The classifier reads exactly 64 frames. Running MediaPipe on more would cost time and change nothing, so longer clips are thinned to 64 evenly spaced frames first. The same rule applies when building the dataset and when predicting, so the model is tested on what it was trained on.
- *Which hand is which.* MediaPipe's own left/right label is a guess from the hand crop and flips when hands cross. We instead match each detected hand to the nearest wrist of the pose skeleton, which sees the whole arm. The label is only a fallback when no body is found.
- *Raw output, no normalisation here.* Extraction is the slow step (about an hour for the dataset on a laptop). Keeping its output raw means the normalisation can be changed later without redoing it.
- *Video mode with a fresh tracker per clip.* Tracking between frames is faster and steadier than detecting from scratch each frame, but the tracker's memory must not carry over from one clip to the next.

### `src/normalize.py` — keypoints to model input

**What it does.** Turns one clip of raw points into a fixed table of 64 rows (frames) by
227 numbers. Per frame: 7 upper-body points, both hands positioned relative to the body,
both hands described in their own frame, and 3 flags saying whether each hand and the
body were found.

**Why it is built this way.**
- *Body frame: centred on the shoulders, measured in shoulder widths.* A sign should look the same wherever the signer stands and however far away they are. Subtracting the shoulder midpoint removes position; dividing by shoulder width removes distance.
- *Hand frame: centred on the wrist, measured in palm lengths.* This describes the handshape alone, independent of where the hand is or how big it appears. It is the "pose-invariant 3D hand feature" the brief asks for. We keep both views because signs differ by handshape *and* by location.
- *Missing hand = zeros plus a flag.* A hand that is hidden or not detected is filled with zeros and its flag set to 0. The model is trained with hands deliberately hidden, so "no hand" is a state it knows rather than nonsense coordinates.
- *Stretch to 64 frames.* Clips have different lengths; the model needs one size. Short clips are stretched by interpolation, which never blends a real point with a missing one.
- *Everything clipped to [-3, 3].* The INT8 model has only 256 input levels. A fixed, known range means those levels are spent where the data is.

### `scripts/extract_all.py` — cache the dataset

**What it does.** Runs `landmarks.py` over every WLASL clip in the vocabulary, in parallel,
and saves one small file per clip. Writes a manifest (one row per clip with its word,
split and status) and a report of usable clips per word and per split.

**Why it is built this way.**
- *Cache once.* MediaPipe takes seconds per clip; training reads the cached files in milliseconds. That is what makes a training run take minutes.
- *Resumable, never crashes on one bad file.* A broken video becomes a row marked as an error, and a rerun skips clips already done.
- *The report is generated, not typed.* The clip counts quoted in the README come from `outputs/extraction_report.md`.

### `scripts/get_mediapipe_models.py`

Downloads the two MediaPipe model files once and prints their sizes, which count toward
the size budget. After this step nothing in the project needs the network.

---

## Person B: model and training

### `src/augment.py` — make the small dataset go further

**What it does.** Produces a slightly different version of each training clip every epoch:
tilts it, stretches it, mirrors it, changes its timing, drops frames, adds small noise,
and hides a hand for part of the clip.

**Why it is built this way.**
- *The dataset is tiny.* About 14 training clips per word. Without augmentation the model simply memorises them.
- *Applied to keypoints, not video.* Changing a few thousand numbers is thousands of times cheaper than changing pixels, so every epoch gets fresh variations for free.
- *Each augmentation mirrors a real-world problem.* Rotation is a tilted camera; mirroring is a left-handed signer; hand dropout is occlusion; frame drops are a stuttering camera. This is also what the robustness test measures.
- *Two were adapted on purpose.* Uniform scaling and uniform speed-up are cancelled exactly by the normalisation, so they would do nothing. We instead stretch x and y by different amounts, and speed up some parts of the sign while slowing others.

### `src/model.py` — the classifier

**What it does.** Reads the 64 x 227 table and outputs a probability for each of the 100
words. It has about 1.59 million parameters.

**Why it is built this way.**
- *Depthwise-separable 1D convolutions* look at a few neighbouring frames to read local motion. "Separable" splits the work into "filter each channel over time" and "mix channels", which gives nearly the same result as a normal convolution for a fraction of the parameters. This is the trick MobileNet uses.
- *Two small self-attention blocks* let any moment of the sign be related to any other, for example the start and end handshape. Convolutions alone only see nearby frames.
- *Built to survive INT8.* BatchNorm rather than LayerNorm, ReLU activations, and attention written from basic operations, because these all have exact integer versions in TFLite.
- *A fallback exists.* `--arch conv` removes the attention blocks, in case attention ever fails to convert. It was not needed.

### `src/train.py` — fit the model

**What it does.** Trains on the train split with fresh augmentation each epoch, checks the
validation split after every epoch, keeps the weights from the best validation epoch, and
only at the very end reports accuracy on the test split.

**Why it is built this way.**
- *Three splits, three jobs.* Train to learn, validation to choose when to stop, test to report. The test split never influences any decision, so the test number is honest.
- *Label smoothing.* The target is "90% this word" rather than "100%". It discourages overconfidence, which matters when there are so few examples.
- *Early stopping.* Training stops when validation accuracy has not improved for a while, and the best weights are restored.
- *Every clip counts at test time.* A test clip where no hand was found still counts in the denominator.

### `src/data.py`

Reads the manifest and the cached landmark files for the other modules. Keeps clip ids as
text, because WLASL ids have leading zeros that a spreadsheet reader would strip.

---

## Person C: quantization, predict, benchmarks, demo

### `src/convert.py` — float model to INT8

**What it does.** Converts the trained Keras model to a full-integer INT8 TFLite file,
checks that no floating-point tensor is left in it, and prints float accuracy next to
INT8 accuracy.

**Why it is built this way.**
- *INT8 means every weight and every intermediate value is an 8-bit integer.* The file is about a quarter of the float size, and integer maths is what phone and wearable chips run fastest and at lowest power.
- *Post-training quantization with a representative set.* To map real numbers onto 256 levels, the converter must know the range each layer produces. It learns this by running 300 real training clips through the model.
- *Strict mode.* The converter is told to fail if any operation lacks an integer version, rather than quietly leaving float operations in.
- *Float and INT8 are compared on the same clips,* so we know exactly what quantization cost.

### `src/classifier.py` — run the INT8 model

**What it does.** Loads the `.tflite` file with the LiteRT interpreter, converts the float
features to int8 on the way in, and converts the int8 output back to probabilities.

**Why it is built this way.**
- *The input scaling lives in one place.* An INT8 model stores a scale and a zero point; the input must be converted with `q = round(x / scale) + zero_point`. A mistake here does not crash, it just gives wrong answers, so one tested function does it for every caller.
- *LiteRT is the runtime named in the brief,* and it is the same engine that runs on Android.

### `predict.py` — the function the evaluator runs

**What it does.** `predict(video_path)` decodes the clip, extracts landmarks, normalises,
classifies with the INT8 model and returns a dictionary with the top word, the top five,
the confidence and the time each stage took.

**Why it is built this way.**
- *Offline by construction.* Every model is loaded from a local file. Nothing is uploaded or downloaded.
- *Each stage is timed separately,* so we can show where the time goes rather than quote a single number.
- *One function owns the output format* (`format_output`). If the organisers' format differs from ours, that one function changes.
- *Models load once* and are reused for every later call.

### `src/benchmark.py` — latency and size

**What it does.** Times 200 runs after warm-up and reports the median and 95th percentile
for the classifier alone, for each stage, per frame, and for a whole clip. Sums the size of
every model file the pipeline loads and compares it with the 20 MB budget.

**Why it is built this way.**
- *p95, not average.* A caption that is usually fast but sometimes stalls is a bad caption; p95 shows the slow cases.
- *Warm-up runs are discarded,* because the first calls include one-off loading costs.
- *All readings of "50 ms" are reported,* because the brief does not say whether it means one classifier call, one frame, or a whole clip.
- *Third-party models are counted in the size.* We report our classifier alone and the full sum, so either interpretation of the budget is covered.

### `src/robustness.py` — what happens when the video gets worse

**What it does.** Makes four damaged copies of every test clip (darkened, rotated,
partly covered, mirrored), runs the full pipeline on each, and reports accuracy against
the undamaged clips.

**Why it is built this way.**
- *The pixels are damaged, not the keypoints.* That way the test includes MediaPipe itself struggling, which is what really happens in poor conditions.
- *Same clips, same frames as the clean run,* so the only difference is the damage.
- *The occluding box is placed by a seed derived from the clip id,* so the result is repeatable.

### `scripts/make_predictions.py` and `scripts/make_results.py`

`make_predictions.py` calls `predict()` on every test clip and writes one JSON line per
clip; a clip that fails still gets a line. `make_results.py` reads that file plus the
benchmark and robustness outputs and writes the six-row `results.csv` with a pass column.
No number in `results.csv` is typed by hand, and the two phone-only metrics are written
as "not measured".

### `scripts/run_all.py`

Runs every step above in order with one command. This is what makes "retrain on a
different word list" a single command: edit `configs/vocab.txt`, run this.

### `demo/live.py` — live captions

**What it does.** Reads the webcam, keeps the last 2.5 seconds of landmarks, classifies
that window a few times per second, and draws the caption on the video.

**Why it is built this way.**
- *Sliding window.* The model was trained on clips of about 2.5 seconds, so the live window is the same length.
- *Confidence threshold.* A word is shown only when the model is confident, so idle moments do not produce random words.
- *Minimum hold time.* Once shown, a word stays for at least 1.5 seconds, so the caption does not flicker between guesses.
- *No hands, no guess.* If hands are barely visible in the window, the classifier is not run at all.

---

## Shared

### `src/config.py` and `configs/vocab.txt`

`config.py` holds every path and constant. `configs/vocab.txt` is the word list, one word
per line, and the line number is the class index. It is the only place the vocabulary is
defined; the model's output size is read from it.

### `tests/`

Small tests for the parts where a mistake would be silent: normalisation (does moving or
resizing the signer leave the features unchanged?), the JSON format, the results scoring
(do failed clips count as wrong?), the caption logic, and the INT8 input scaling.
