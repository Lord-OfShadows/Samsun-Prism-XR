# Assumptions and decisions

Choices made without the team in the room during the Core build (17 Nov 2026). Each has
one line of reasoning. Anything marked **CHECK** should be confirmed by a human.

## Inputs that were missing

| # | Situation | Decision | Why |
|---|---|---|---|
| 1 | `docs/PRD.md` does not exist; the PRD is a PDF in `docs/`. | Read the PDF and treated it as the source of truth. | Same document, different format. |
| 2 | `docs/BRIEF_FORMATS.md` does not exist, so the organisers' exact JSON and CSV formats are unknown. **CHECK** | Chose a schema (below) and isolated it: `format_output()` in `predict.py` for the JSON, `FIELDS` plus the row builders in `scripts/make_results.py` for the CSV. | When the real format arrives, two small places change and nothing else. |
| 3 | No WLASL data in the repo. | Downloaded the WLASL100 videos from a public Hugging Face mirror (`aipieces/WLASL100`, 2,038 clips, 256x256, pre-trimmed) and `WLASL_v0.3.json` from the official WLASL GitHub repo. | All 2,038 clips of the first 100 glosses were present and the official split sizes matched (1,442 / 338 / 258). **CHECK** the mirror's licence terms before redistributing anything; `data/` is git-ignored. |
| 4 | PRD asks for Python 3.10 or 3.11; the existing `.venv` was Python 3.13. | Verified and ran everything in a fresh Python 3.11.18 venv. Left `.venv` untouched. | Keras 3.15 needs 3.11 or newer, so 3.10 is not possible with these pins. |

## Output formats (assumed, see #2)

`predict()` returns:

```json
{"clip_id": "07085", "prediction": "book", "confidence": 0.83,
 "top5": [{"word": "book", "confidence": 0.83}, "... 4 more ..."],
 "latency_ms": {"decode": 0, "landmarks": 0, "normalize": 0, "classifier": 0, "total": 0},
 "frames": 64}
```

- `predictions.jsonl`: one such object per line, one line per clip in the test manifest. A clip that cannot be read still gets a line, with `"prediction": null` and an `error` field.
- `results.csv`: columns `metric,target,value,unit,pass,notes`, exactly the six metrics of the organisers' table. `pass` is `PASS`, `FAIL` or `not measured`.

## Design decisions

| # | Decision | Why |
|---|---|---|
| 5 | Clips longer than 64 frames are subsampled to 64 **before** MediaPipe, both when caching and in `predict()`. | The classifier only sees 64 frames; landmarking more is pure latency. Same sampling in training and inference. |
| 6 | Pose features use nose + shoulders, elbows, wrists (7 points, x and y only). | PRD lists shoulders, elbows, wrists; the nose is added because many signs differ only by where the hand is relative to the face. Pose z is too noisy to help. |
| 7 | One body centre and scale per clip (median over frames), not per frame. | A signer does not move or change distance within a two-second sign; per-frame values would add shoulder jitter to every hand coordinate. |
| 8 | Each hand appears twice: in the body frame (where it is) and in its own wrist-centred, palm-scaled 3D frame (what shape it makes). | The PRD says hands are "also" expressed in their own frame; location and handshape are both needed to tell signs apart. |
| 9 | All features are clipped to [-3, 3]. | Gives the INT8 input one fixed, known range (about 0.024 per step). |
| 10 | If no pose is found in a whole clip, the body frame falls back to the image centre and a quarter of the frame width. | Keeps `predict()` returning a valid answer instead of failing. |
| 11 | "Time stretching" is implemented as a random trim of the ends plus a non-uniform speed warp. | A uniform speed change is cancelled exactly by resampling to 64 frames, so it would do nothing. |
| 12 | "Scaling" augmentation stretches x and y by different amounts. | Uniform scaling is cancelled exactly by shoulder-width normalisation. |
| 13 | "Whole-hand dropout" hides a hand for a contiguous 10 to 50% of the clip, never the whole clip. | Hiding the only signing hand for the whole clip would make the label unrecoverable. |
| 14 | Label smoothing 0.1, dropout 0.2 in blocks and 0.4 before the output layer, AdamW with cosine schedule, early stopping on validation top-1. | PRD names the techniques but not the values; these are common defaults. Not tuned. |
| 15 | BatchNorm instead of LayerNorm in the attention blocks; attention written with Dense, matmul and softmax. | These convert to integer TFLite ops cleanly; LayerNorm is known to quantise badly. |
| 16 | The model is selected on the validation split only; the test split is read once, for the final number. Training uses the train split only (not train + val). | Keeps the test number honest. Some papers train on train + val, so their numbers are not directly comparable. |
| 17 | Test accuracy is computed over **all** clips in the test split; a clip that fails to decode or has no hands detected counts as a miss. | The denominator must not shrink to flatter the number. |
| 18 | `predict(video_path)` uses the whole file. WLASL frame bounds are applied when caching training data (only where the file is longer than the bounds). | The evaluator's signature has only a path. A few clips whose file is longer than their annotated bounds are therefore seen whole at test time. |
| 19 | Robustness perturbs the pixels and re-runs MediaPipe: brightness x0.25, 15 degree rotation, a grey box over 35% x 35% of the frame at a per-clip seeded position in the chest and hands region, and a left-right flip. The reported drop is clean top-1 minus the mean top-1 of the four sets. | PRD names the four conditions but not their strength. **CHECK** these against the organisers' definition if one exists. |
| 20 | `results.csv` has one latency row. It holds the **whole-clip, end-to-end** p95 (the strictest reading of "50 ms"); the classifier-only and per-frame p95 are in the notes column. Switch with `LATENCY_METRIC_FOR_RESULTS` in `src/config.py`. **CHECK** | The brief's definition is an open question in the PRD. The strictest reading cannot be accused of flattering us. |
| 21 | `model_size` pass/fail is judged against 20 MB although the results table says 25 MB. Size is reported in MB = 1,000,000 bytes. | PRD: "We design for 20 MB." Decimal MB gives the larger, more conservative number. |
| 22 | The MediaPipe model files are downloaded by `scripts/get_mediapipe_models.py` at setup and are not committed (the existing `.gitignore` excludes them). After setup, nothing uses the network. | Respects the existing repo decision. **CHECK**: if the evaluator's machine has no internet at all, commit the two `.task` files (13.6 MB). |
| 23 | Live demo: 2.5 s sliding window, re-classified every 0.3 s, caption threshold 0.6, hold 1.5 s; no classification when a hand is visible in under 30% of the window. | WLASL clips have a median of 61 frames at 25 fps, so 2.5 s matches the training clips. Threshold and hold are starting values in `src/config.py`, to be tuned by eye at the webcam. |
| 24 | Float-vs-INT8 accuracy is printed by `src/convert.py` even though the comparison table is a Target-tier item. | It is the only way to verify the INT8 export works; no extra tooling was built for it. |
| 25 | Commits carry no AI co-author trailer. AI assistance is disclosed in the README instead. | Instruction from the repo owner during the session. |
