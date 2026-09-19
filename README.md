# Hand-Movement Head-Touch Detection

**V1 goal:** given a continuous video, automatically identify the moments when a hand
touches the head and return their timestamps.

We are building this incrementally, starting with a simple, explainable computer-vision
baseline (no ML training) before considering anything more complex.

## Development phases

1. **Data Collection / Video Understanding** — done (this document)
2. **Ground Truth / Annotation** — done, 3 events annotated in `test_video_task_annotations.csv`
3. **Core Detection** — done (MediaPipe hand + pose landmarks, normalized spatial distance)
4. **Event Detection + Timestamping** — done (debounced threshold state machine, evaluated against ground truth)

All four phases are implemented.

## Project structure

```
Hand-movement-detection-proj/
├── data/
│   ├── test_video_task.mp4   # source recording (never modified)
│   └── Old/
│       └── test_video.mp4    # earlier, longer recording — superseded, kept for reference
├── src/
│   ├── inspect_video.py      # Phase 1: report video metadata, sample frames
│   ├── annotate.py           # Phase 2: interactive ground-truth annotation tool
│   ├── timeutils.py          # shared MM:SS.ss <-> seconds helpers
│   ├── download_models.py    # fetches the MediaPipe model bundles into models/
│   ├── head_touch_detector.py# Phase 3+4: landmark extraction, distance signal, event extraction
│   ├── evaluate.py           # Phase 4: match detected events to ground truth, report metrics
│   └── tune_threshold.py     # grid-search thresholds against a cached signal + ground truth
├── models/                   # MediaPipe .task model bundles, gitignored (run download_models.py)
├── annotations/
│   ├── annotation_format_example.csv   # format reference, not real data
│   └── test_video_task_annotations.csv # real ground truth (3 events)
├── outputs/                  # generated artifacts (sample frames, detected events, etc.), gitignored
├── .venv/                    # local virtual environment, gitignored
├── requirements.txt
└── README.md
```

## Environment setup

A local virtual environment (`.venv`) has already been created with Python 3.14.

```powershell
# create (already done)
python -m venv .venv

# activate
.venv\Scripts\Activate.ps1        # PowerShell
.venv\Scripts\activate.bat        # cmd

# install dependencies
pip install -r requirements.txt
```

**Dependencies installed so far:** `opencv-python`, `numpy`. That's all Phase 1/2 needs.

`pandas` and `mediapipe` are intentionally **not** installed yet — they belong to Phase 3
(landmark detection) and beyond. `mediapipe` was checked and does publish a
Python-version-agnostic wheel (`py3-none-win_amd64`) compatible with this environment, so
there's no blocker when we get there.

## Video properties (Phase 1)

Captured with `src/inspect_video.py data/test_video_task.mp4`:

| Property   | Value |
|------------|-------|
| filename   | test_video_task.mp4 |
| size       | 345.87 MB |
| resolution | 1280 x 720 |
| fps        | 30.0 |
| frame count| 9,349 |
| duration   | 311.6 s (≈ 5 min 12 s) |
| codec      | H.264 |

This supersedes the original, longer recording (`data/Old/test_video.mp4`, ≈8 min 19 s),
which was trimmed down and re-exported. Everything below refers to the current video.

Run it yourself any time:

```powershell
.venv\Scripts\python.exe src\inspect_video.py data\test_video_task.mp4
# optionally sample frames to eyeball the recording:
.venv\Scripts\python.exe src\inspect_video.py data\test_video_task.mp4 --sample-frames 6
```

## Ground-truth annotation (Phase 2)

### Format

CSV with one row per head-touch event:

```
event_id,start_time,contact_time,end_time,hand,notes
1,00:31.20,00:31.47,00:32.10,right,"touches forehead"
```

- `start_time` — hand begins moving toward the head
- `contact_time` — the moment of actual contact
- `end_time` — hand leaves the head
- `hand` — `left` / `right` / `both`
- `notes` — free text (e.g. "scratches head", "adjusts glasses", "false-positive-prone: hand passes near ear")

Timestamps use `MM:SS.ss` (minutes:seconds, matching `inspect_video.py`'s output). The
video is under an hour long so minutes-only is sufficient; if that ever changes, extend
the format to `HH:MM:SS.ss`.

A reference example lives in `annotations/annotation_format_example.csv` — it is not real
data, just a schema illustration. Real ground truth goes in
`annotations/test_video_task_annotations.csv`, created the first time you save an event.

### Annotation tool

`src/annotate.py` opens the video in an OpenCV window and lets you scrub through it and
log events without hand-timing anything in a separate player.

```powershell
.venv\Scripts\python.exe src\annotate.py data\test_video_task.mp4 annotations\test_video_task_annotations.csv
```

Controls:

| Key | Action |
|-----|--------|
| space | play / pause |
| a / d | step one frame back / forward (while paused) |
| j / l | jump ~1 second back / forward (while paused) |
| s | mark **start** of an event at the current timestamp |
| c | mark **contact** (moment of touch) at the current timestamp |
| e | mark **end**, then prompts in the console for `hand` and `notes`, and appends the row to the CSV |
| r | discard pending start/contact marks without saving |
| q / Esc | quit (rows already saved remain in the CSV) |

Each saved event is appended to the CSV immediately, so quitting early never loses
already-logged events. `event_id` continues from whatever is already in the file, so you
can stop and resume across multiple sessions.

## Core detection (Phase 3 + 4)

### Method

A hand-crafted geometric baseline — no model training, consistent with the project's
"explainable first" approach.

1. **Landmarks: MediaPipe Tasks API.** Per frame, run the **Hand Landmarker** (21
   keypoints/hand + left/right handedness) and the **Pose Landmarker** (33 body
   keypoints), both in `VIDEO` running mode.
   - Pose is used for the head instead of Face Mesh **on purpose**: the moment a hand
     touches the head is exactly when the face becomes partially occluded, and the pose
     model degrades far more gracefully under partial occlusion than a dense face mesh
     does — we need landmarks to stay stable at exactly the moment we care about most.
2. **Head as a circle.** Approximate the head with a center (visibility-weighted
   centroid of nose/eyes/ears) and a radius, `head_scale`, estimated from whichever of
   these is available, in priority order: ear-to-ear distance, else 2.6× eye-to-eye
   distance, else 0.55× shoulder-to-shoulder distance. These are standard anthropometric
   proxies for head width; the fallback chain keeps an estimate available even when the
   head is turned or partly out of frame.
3. **Per-frame distance signal.** For each detected hand, take the minimum pixel
   distance from its fingertip + wrist landmarks to the head center, normalized by
   `head_scale`. A value ≲ 1.0 means a hand landmark has entered the head disk. This is
   scale-invariant (robust to the subject moving closer/further from the camera).
4. **Speed gate.** A frame only counts as touching if the hand is close **and slow**. A
   real touch decelerates to a near standstill at the head; a gesture or hair-flick
   passes through the close zone at speed. Speed is the **wrist's** pixel speed
   (a stable anchor — the "closest fingertip" used for distance can flip between fingers
   and fake motion), normalized by `head_scale` to head-widths/sec and smoothed over 3
   frames. Gaps in hand detection longer than 0.25 s make speed "unknown" (not slow).
5. **Event extraction.** A small debounce state machine turns the noisy per-frame signal
   into discrete events: a few consecutive close-and-slow frames confirm a touch start
   (so `--enter-frames` doubles as the dwell-time requirement),
   a few consecutive frames failing the gate confirm the end (so single-frame landmark
   jitter can't fabricate an event), and a frame where the hand briefly isn't detected
   *while already touching* is treated as still-touching rather than an automatic end
   (self-occlusion from the hand covering the head is expected exactly then).
   `contact_time` is reported as the frame of minimum normalized distance within the
   first 1.5 s of the event (bounded because a hand lingering near the head can keep one
   event open for many seconds, and a global minimum could then pick an unrelated later dip).
6. **Evaluation.** Detected events are matched to ground-truth events by nearest
   `contact_time`, one-to-one, within a **±0.5 s** tolerance window — generous enough to
   absorb landmark noise while still meaning something, chosen against a ground truth of
   only 3 events. Hand label (left/right) is reported per match as a diagnostic but does
   **not** gate the match, since MediaPipe's handedness convention can come out mirrored
   relative to the real-world hand depending on how the source video was captured.

### Running it

```powershell
# one-time: fetch the MediaPipe model bundles (~13 MB, gitignored)
.venv\Scripts\python.exe src\download_models.py

# run detection (several minutes for a 5-minute video on CPU — pose + hand
# inference per frame is the bottleneck, not I/O)
.venv\Scripts\python.exe src\head_touch_detector.py data\test_video_task.mp4 `
    outputs\detected_events.csv --signal-csv outputs\distance_signal.csv

# compare against ground truth
.venv\Scripts\python.exe src\evaluate.py annotations\test_video_task_annotations.csv `
    outputs\detected_events.csv --tolerance 0.5
```

`--threshold`, `--velocity-threshold`, `--enter-frames`, and `--exit-frames` on
`head_touch_detector.py` control the gate and state machine; the defaults
(`1.0`, `0.5`, `3`, `10`) came from `src/tune_threshold.py`. Because `--signal-csv`
caches the raw per-frame distances and wrist positions, retuning never needs the
(expensive) landmark extraction again:

```powershell
.venv\Scripts\python.exe src\tune_threshold.py outputs\distance_signal.csv `
    annotations\test_video_task_annotations.csv
.venv\Scripts\python.exe src\head_touch_detector.py data\test_video_task.mp4 `
    outputs\detected_events.csv --from-signal outputs\distance_signal.csv
```

### Results

Run against `data/test_video_task.mp4` (5:12, 9,349 frames) with the tuned defaults
(`--threshold 1.0 --velocity-threshold 0.5 --enter-frames 3 --exit-frames 10`, picked via
`src/tune_threshold.py` grid-searching against ground truth):

| Metric | Distance only | + speed gate (current) |
|---|---|---|
| Ground-truth events | 3 | 3 |
| Detected events | 24 | 11 |
| True positives | 3 | 3 |
| False negatives (missed) | 0 | 0 |
| False positives (extra) | 21 | 8 |
| **Recall** | **1.00** | **1.00** |
| **Precision** | 0.12 | **0.27** |
| F1 | 0.22 | 0.43 |
| Mean timing error on matches | 0.20 s | 0.26 s |

All 3 real head-touches are found, with `contact_time` within 0.03–0.46 s of the
annotated moment (inside the ±0.5 s tolerance, though one match sits near its edge). The
speed gate cut false positives from 21 to 8 without losing a real touch; the 0.5
head-widths/sec setting held across neighbouring enter/exit values rather than being a
one-off spike. Precision is still low: the remaining detections are moments where a hand
slows down near the head without an annotated touch (adjusting hair, resting a hand near
the ear/chin, etc.), which the head-as-a-circle model can't tell apart from a real
touch. Some have a very small `min_normalized_distance` and are plausibly genuine brief
contacts that weren't logged as "head touches" — a labeling-scope question as much as a
detector error.

**Why recall was favored over precision:** for a system meant to *find* head-touch
events, missing a real one is worse than flagging an extra candidate a human can quickly
rule out.

**Caveat — small-sample tuning.** Both thresholds were tuned against only 3 events from
one person in one camera setup, so treat 0.27 precision as an in-sample number, not a
generalization estimate. Validating on other people, cameras and lighting is the real
next step, and it is what would decide whether a light trained classifier over these same
features (distance, speed, dwell) is worth adding.

## Design decisions

- **No pandas.** All CSV reading/writing (annotations, detected events, the raw distance
  signal) goes through the stdlib `csv` module — the data is small and simple enough
  that pandas would be unused surface area.
- **MediaPipe Tasks API, not the older `mp.solutions.*` API.** The legacy solutions API
  is deprecated in current `mediapipe` releases; `mediapipe.tasks.python.vision` is the
  supported path and is what a fresh `pip install mediapipe` gives you.
- **Pose landmarks for the head, not Face Mesh** — see the occlusion-robustness reasoning
  in the Method section above.
- **CSV over a heavier annotation format** (e.g. JSON, a database) — keeps ground truth
  human-editable and diffable, and trivial to load later with `csv` or `pandas`.
- **Timestamps as `MM:SS.ss` strings**, not raw seconds — easier to eyeball and matches
  what most video players show. `src/timeutils.py` centralizes the `MM:SS.ss <-> seconds`
  conversion so every script (annotate, inspect, detect, evaluate) agrees on the format.
- **`outputs/` and `models/` are gitignored** — regenerated artifacts and large downloaded
  binaries, not source-of-truth data. `src/download_models.py` makes the model bundles
  reproducible for a fresh clone.
