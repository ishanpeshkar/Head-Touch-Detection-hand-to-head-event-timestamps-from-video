# Hand-Movement Head-Touch Detection

**V1 goal:** given a continuous video, automatically identify the moments when a hand
touches the head and return their timestamps.

We are building this incrementally, starting with a simple, explainable computer-vision
baseline (no ML training) before considering anything more complex.

## Development phases

1. **Data Collection / Video Understanding** — done (this document)
2. **Ground Truth / Annotation** — done, 5 events annotated in `test_video_task_annotations.csv`
3. **Core Detection** — done (MediaPipe hand + pose landmarks, normalized spatial distance)
4. **Event Detection + Timestamping** — done (debounced threshold state machine, evaluated against ground truth)

All four phases are implemented.

## Project structure

```
Hand-movement-detection-proj/
├── data/                     # NOT in the repo (gitignored, 345 MB video) — see "Reproducing"
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
│   ├── tune_threshold.py     # grid-search thresholds against a cached signal + ground truth
│   └── visualize_events.py   # draws the detector's view on each detected contact frame
├── docs/
│   └── how-it-works.md       # detailed walkthrough of the method, with example frames
├── results/                  # committed sample output: detected events, signal, annotated frames
│   ├── detected_events.csv
│   ├── distance_signal.csv
│   └── frames/               # one frame per detected event (4 TP_*, 7 FP_*, overlap rule) plus 1 FN_* for the missed touch
├── data_example/             # placeholder marking where the video goes (real video not in repo)
├── models_example/           # placeholder marking where the model bundles go
├── models/                   # MediaPipe .task model bundles, gitignored (run download_models.py)
├── annotations/
│   ├── annotation_format_example.csv   # format reference, not real data
│   └── test_video_task_annotations.csv # real ground truth (5 events)
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
# open near a known moment instead of scrubbing from the start:
.venv\Scripts\python.exe src\annotate.py data\test_video_task.mp4 annotations\test_video_task_annotations.csv --start 02:30.00
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

A longer walkthrough with example frames is in [docs/how-it-works.md](docs/how-it-works.md).
Sample output (detected events and annotated frames) is committed under
[results/](results/) so it can be inspected without running the pipeline.

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
   only 3-5 events. Hand label (left/right) is reported per match as a diagnostic but does
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
`src/tune_threshold.py` grid-searching against ground truth).

**Ground truth: 5 annotated touches, built up in three steps.**
1. The original annotation had 3 touches. **The thresholds were tuned on these 3 only.**
2. A 4th (right hand flat on the forehead, 02:34.03 / 02:34.90 / 02:36.10) was found during
   the false-positive review below and added.
3. A 5th (right hand, a quick touch, 03:17.70 / 03:18.13 / 03:18.60) was noticed afterwards
   and added. The detector does not find it.

**Neither addition triggered a re-tune.** Two of the five touches were therefore added
after the detector's output had been seen; see the limitations.

Two matching rules are reported. The **strict** rule is the headline: a detection matches
if its `contact_time` is within ±0.5 s of the annotated `contact_time`. The **overlap**
rule additionally accepts a detection whose `[start, end]` interval overlaps the annotated
one. It was added *after* seeing that the strict rule scores the 4th touch as a miss, so
treat it as a secondary, more lenient number, not a replacement.

| 5 annotated touches | Distance only, strict | **Speed gate, strict** | Distance only, overlap | Speed gate, overlap |
|---|---|---|---|---|
| Detected events | 24 | 11 | 24 | 11 |
| True positives | 4 | 3 | 5 | 4 |
| False negatives (missed) | 1 | 2 | 0 | 1 |
| False positives (extra) | 20 | 8 | 19 | 7 |
| **Recall** | 0.80 | **0.60** | 1.00 | 0.80 |
| **Precision** | 0.17 | **0.27** | 0.21 | 0.36 |
| F1 | 0.28 | 0.37 | 0.34 | 0.50 |
| Mean timing error on matches | 0.22 s | 0.26 s | 0.35 s | 0.42 s |

**The speed gate is a trade-off, not a free improvement.** It cut extra detections from 20
to 8 (strict), but it lowers recall from 0.80 to 0.60, because it rejects the 5th touch,
which the distance-only version does find. That touch is quick: while the hand is close to
the head (distance 0.77–0.96, from about 03:18.03 to 03:18.47) its wrist speed is
0.74–3.55 head-widths/sec (mostly 0.7–1.5), always above the 0.5 threshold, so no frame
counts as slow. The
gate assumes a touch is a hand coming to rest; a brief tap does not.
Separately, 58 (frame, hand) pairs in the signal contain two detections with the *same*
hand label (MediaPipe labelled both hands "Right" in that frame), which interleaves two
hands in the per-hand speed calculation and produced unknown speeds at the start of this
touch. That is a bug in how hand identity is handled, not fixed here.

The 4th touch is *detected* (detection #5) but scores as a miss under the strict rule. The
hand lands at about 02:34.9 and stays on the forehead until about 02:36.1. The annotation
marks the landing, while the detector's `contact_time` is the frame where the hand is
*closest*, which here is 02:35.80, 0.90 s later. The same lag shows in two of the other
detected touches (+0.30 s and +0.46 s; the third is +0.03 s). Precision is still low: see
the false-positive review below for what the extra detections are.

### Review of the 8 false positives

Each extra detection under the strict rule was inspected by viewing the frame at its
contact time (`results/frames/`, generated by `src/visualize_events.py`; the frames are
labelled with the overlap rule, so detection #5 appears as `TP_det05`). This is
one frame per event, reviewed by eye, not a re-annotation of the video. The review is
what surfaced the missed forehead touch (detection #5), which was then added to the ground
truth. The 5th touch (03:18) has no detection, so it appears as `FN_gt05_03m18.13s.jpg`,
a frame at the annotated contact time labelled as a miss, not as a true positive.

| Det | Time | Hand | What the frame shows | Verdict |
|---|---|---|---|---|
| 2 | 00:47.77 | both | Both open hands raised beside the face, thumbs inside the circle | False positive |
| 5 | 02:35.80 | right | Right hand flat on the forehead | **Real head touch; now annotated as GT#4** (a miss under the strict rule only because its contact time is 0.90 s after the annotated landing) |
| 6 | 03:14.00 | both | Both open hands raised beside the face | False positive |
| 7 | 03:14.80 | both | Both open hands raised beside the face | False positive |
| 8 | 03:16.60 | both | Both open hands raised beside the face | False positive |
| 9 | 03:20.13 | right | Open hand raised at the side of the face (waving), thumb at the circle edge | False positive |
| 10 | 03:47.80 | left | Hand on the chin / jaw, covering the lower face | Face contact, not annotated |
| 11 | 05:06.47 | right | Hand over the eye (rubbing / covering the eye) | Face contact, not annotated |

**Findings**

- **5 of 8 are the same failure:** an open hand raised beside the face. The thumb tip
  falls inside the head circle although nothing touches the head. The circle is the cause:
  its radius is the full ear-to-ear width, so it extends well beyond the actual head, and
  it is centred at eye level.
- **3 of 8 are genuine hand-on-face contacts.** #5 (forehead) is a real head touch and is
  now in the ground truth. #10 (chin/jaw) and #11 (eye) are contacts with the lower face
  and eye, which the ground truth deliberately does **not** count: it covers contact with
  the head or forehead, not the lower face or eyes. Counting those two as well, on top of
  the 4 overlap-rule true positives, would give 6 of 11 correct (precision about 0.55) — indicative only, since it depends on that scope
  decision and on one person's reading of single frames.
- **The same geometry limits recall-friendly settings.** In two of the true
  positives (ground-truth touches #1 and #3) the fingertips resting on the top of the head lie *outside* the
  circle, and detection fires because the *wrist* is inside it. Shrinking the circle to
  remove the open-hand false positives would therefore also lose those touches unless the
  circle is moved up toward the top of the head. This was not tried; it is the first
  improvement to make.

**Why recall was favored over precision:** for a system meant to *find* head-touch
events, missing a real one is worse than flagging an extra candidate a human can quickly
rule out. That was the aim when tuning on the original 3 touches; on the 5-touch ground
truth the speed gate actually lowers recall (0.80 to 0.60 strict), which is the opposite
of that aim and the first thing to revisit.

**Caveat — small-sample tuning.** Both thresholds were tuned against only the original 3
events from one person in one camera setup, so treat 0.27 (strict) / 0.36 (overlap)
precision as in-sample numbers, not a generalization estimate. Validating on other people,
cameras and lighting is the real next step, and it is what would decide whether a light trained classifier over these same
features (distance, speed, dwell) is worth adding.

### Worked example

Ground-truth touch #3 is annotated with contact at **02:27.93**. The detector reports
start **02:27.63**, contact **02:27.90**, min normalized distance **0.571**, i.e. a
fingertip landed well inside the head circle (< 1.0) while the wrist had slowed below
0.5 head-widths/sec for 3+ consecutive frames. Timing error: **0.03 s**.

## Limitations and honest notes

- **What this is.** A rule-based detector on top of *pretrained* MediaPipe landmark
  models. Nothing here is trained on this project's data; the annotations are used only to
  evaluate the detector and to tune four numbers (distance threshold, speed threshold,
  enter/exit frame counts).
- **Results are in-sample.** Those four numbers were tuned against the original 3 events
  from one person, one camera and one recording, and not re-tuned after a 4th and 5th were
  added. The reported precision/recall describe that recording, not how the system would do
  on new footage. With 5 events, one event more or less moves recall by 20 points, and one
  match (+0.46 s) sits near the ±0.5 s tolerance edge.
- **The ground truth changed after the detector ran.** Two touches were added afterwards:
  the 4th was found through the false-positive review, the 5th was noticed later. These
  are corrections of annotation errors, not tuning, and the tuning was not repeated, but
  the ground truth is not fully independent of the detector's output. The lenient overlap
  rule was likewise added after seeing results.
- **The speed gate misses quick touches.** A brief tap that never comes to rest (the 5th
  touch) stays above the speed threshold and is rejected; the distance-only version finds
  it. The gate trades recall for precision.
- **Hand identity is just the MediaPipe label.** In 58 (frame, hand) pairs both hands get
  the same label, which merges two hands into one track for speed and event extraction.
- **Precision is low (0.27 strict).** The 8 false positives were reviewed by eye (see "Review of the 8
  false positives"): 5 are open hands raised beside the face, 1 is the missed forehead
  touch, 2 are real contacts with the lower face / eye that the ground truth does not
  count. That review is one frame per event by one reviewer.
- **Detected `contact_time` lags the landing.** It is the frame where the hand is closest,
  while the annotations mark where it lands, so for a hand that lands and holds it can be
  a second or more later. Lag on the four detected touches: +0.30, +0.46, +0.03 and
  +0.90 s (the 5th touch is not detected).
- **Pose vs. Face Mesh was reasoned, not tested.** Pose landmarks were chosen for the
  head because face tracking is expected to degrade when a hand covers the face. The two
  were not compared empirically on this video.
- **Distance + speed can't see contact.** The head is a circle inferred from a few
  keypoints; the detector knows "a hand landmark is inside the circle and the wrist is
  slow", not "skin/hair is touched". The circle is larger than the head (radius = full
  ear-to-ear width) and centred at eye level, so it both admits nearby open hands and
  misses the top of the head. It is also 2D, so a hand held *in front of* the head
  toward the camera looks identical to one touching it.
- **`contact_time` ignores speed.** It is the closest frame within 1.5 s of the start, so
  the hand can still be moving at that frame (about 2.3 head-widths/sec at the contact
  frame of the third touch).
- **Coverage gaps.** Only 3,407 (frame, hand) rows exist for 9,349 frames: when no hand
  or no usable head estimate is found the frame produces no row, so touches during
  detection dropouts can be missed.
- **Hand labels happened to match.** MediaPipe's left/right labels agreed with the
  annotations on every matched touch for this video, so no mirroring correction was needed;
  this may not hold for other cameras (e.g. mirrored webcams). Matching therefore ignores
  the label and reports it as a diagnostic.
- **Toward production.** Validate on multiple people/cameras/lighting first. If precision
  is still inadequate, train a small classifier over the *existing* features (distance,
  speed, dwell time) rather than end-to-end video models; both need labeled footage
  beyond this one recording.

## Reproducing

The source video is a personal recording and is **not shared** (it is also 345 MB, over
GitHub's 100 MB file limit), and the model bundles are not in the repo. The committed
`results/` folder holds the detected events, the per-frame signal and annotated frames from
the run reported here, and `annotations/` holds the ground truth, so the evaluation step
below can be re-run without the video. Re-running the detection itself needs a video: put
one at `data/test_video_task.mp4`, or point the commands at your own video (the annotation
tool in `src/annotate.py` creates matching ground truth for it):

```powershell
pip install -r requirements.txt
.venv\Scripts\python.exe src\download_models.py
.venv\Scripts\python.exe src\head_touch_detector.py data\test_video_task.mp4 `
    outputs\detected_events.csv --signal-csv outputs\distance_signal.csv   # ~14 min on CPU
.venv\Scripts\python.exe src\evaluate.py annotations\test_video_task_annotations.csv `
    outputs\detected_events.csv
```

## Design decisions

- **No pandas.** All CSV reading/writing (annotations, detected events, the raw distance
  signal) goes through the stdlib `csv` module — the data is small and simple enough
  that pandas would be unused surface area.
- **MediaPipe Tasks API, not the older `mp.solutions.*` API.** The legacy solutions API
  is deprecated in current `mediapipe` releases; `mediapipe.tasks.python.vision` is the
  supported path and is what a fresh `pip install mediapipe` gives you.
- **Pose landmarks for the head, not Face Mesh** — see the occlusion-robustness reasoning
  in the Method section above (a hypothesis; not compared empirically, see Limitations).
- **CSV over a heavier annotation format** (e.g. JSON, a database) — keeps ground truth
  human-editable and diffable, and trivial to load later with `csv` or `pandas`.
- **Timestamps as `MM:SS.ss` strings**, not raw seconds — easier to eyeball and matches
  what most video players show. `src/timeutils.py` centralizes the `MM:SS.ss <-> seconds`
  conversion so every script (annotate, inspect, detect, evaluate) agrees on the format.
- **`outputs/` and `models/` are gitignored** — regenerated artifacts and large downloaded
  binaries, not source-of-truth data. `src/download_models.py` makes the model bundles
  reproducible for a fresh clone.
