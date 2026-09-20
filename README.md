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
│   ├── tune_threshold.py     # grid-search gate settings against a cached signal + ground truth
│   ├── geometry_experiment.py# Pass 6: compare head shapes offline, incl. leave-one-out
│   ├── visualize_events.py   # draws the detector's view on each detected contact frame
│   ├── live_demo.py          # live webcam / stream mode: same pipeline, frame by frame
│   └── reencode_h264.py      # dev helper: make a --record video browser-playable (H.264)
├── tests/
│   └── test_touch_logic.py   # event-logic tests: streaming == original batch logic, synthetic cases
├── demo/                     # two short annotated clips run through the live pipeline
├── docs/
│   └── how-it-works.md       # detailed walkthrough of the method, with example frames
├── results/                  # committed sample output: detected events, signal, annotated frames
│   ├── detected_events.csv
│   ├── distance_signal.csv   # per-frame signal incl. raw geometry, so head shapes can be re-tested
│   ├── geometry_experiment.txt # output of the Pass 6 head-shape experiment
│   ├── frames/               # current (Pass 6) frames: one per detected event, TP_* / FP_*
│   └── frames_pass5/         # original-head-model frames incl. the raised-hand false positives and the missed touch
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

**Dependencies** (`requirements.txt`): `opencv-python`, `numpy` and `mediapipe`.
`mediapipe` publishes a Python-version-agnostic wheel (`py3-none-win_amd64`) that installs
cleanly on Python 3.14. `pandas` is intentionally not used (see Design decisions).

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
   - Pose is used for the head instead of Face Mesh because the moment a hand touches the
     head is when the face becomes partially occluded, and the pose model is expected to
     degrade more gracefully under partial occlusion than a dense face mesh. **This was
     reasoned, not tested.**
2. **Head zone.** A circle over the top of the head, sized from a head-width estimate
   `head_scale` (ear-to-ear distance, else 2.6× eye-to-eye, else 0.55× shoulder width).
   Its centre is the visibility-weighted centroid of nose/eyes/ears **raised by 0.6 ×
   `head_scale`**, and its radius is **0.6 × `head_scale`**. (The original model — centred
   at eye level, radius = full head width — is `--head-up 0 --head-radius 1`; see Pass 6.)
3. **Per-frame distance signal.** For each detected hand, take the minimum pixel distance
   from its fingertip + wrist landmarks to the zone's centre, divided by the radius. A
   value ≲ 1.0 means a hand landmark is inside the zone. Everything is in head-widths, so
   it is scale-invariant (robust to the subject moving closer/further from the camera).
4. **Speed gate.** A frame only counts as touching if the hand is inside the zone **and**
   not moving too fast (a hand resting on the head is slow; one sweeping through is fast).
   Speed is the **wrist's** pixel speed (a stable anchor — the "closest fingertip" used for
   distance can flip between fingers and fake motion), in head-widths/sec, smoothed over 3
   frames, threshold **1.5**. Gaps in hand detection longer than 0.25 s make speed
   "unknown" (not slow).
5. **Event extraction.** A small debounce state machine turns the noisy per-frame signal
   into discrete events: a few consecutive close-and-slow frames confirm a touch start
   (so `--enter-frames` doubles as the dwell-time requirement), a few consecutive frames
   failing the gate confirm the end (so single-frame landmark jitter can't fabricate an
   event), and a frame where the hand briefly isn't detected *while already touching* is
   treated as still-touching rather than an automatic end. `contact_time` is the frame of
   minimum normalized distance within the first 1.5 s of the event (bounded because a hand
   lingering near the head can keep one event open for many seconds, and a global minimum
   could then pick an unrelated later dip). It is the *closest* frame, not the moment of
   landing, so it lags the annotated landing.
6. **Evaluation.** Detected events are matched to ground-truth events one-to-one by
   nearest `contact_time` within **±0.5 s** (the **strict** rule, the headline). A second,
   more lenient **overlap** rule also accepts a detection whose `[start, end]` interval
   overlaps the annotated one; it was added after seeing results (see below). Hand label
   (left/right) is reported per match as a diagnostic but does **not** gate the match,
   since MediaPipe's handedness convention can come out mirrored depending on how the
   source video was captured.

### Running it

```powershell
# one-time: fetch the MediaPipe model bundles (~13 MB, gitignored)
.venv\Scripts\python.exe src\download_models.py

# run detection (several minutes for a 5-minute video on CPU — pose + hand
# inference per frame is the bottleneck, not I/O)
.venv\Scripts\python.exe src\head_touch_detector.py data\test_video_task.mp4 `
    outputs\detected_events.csv --signal-csv outputs\distance_signal.csv

# compare against ground truth (add --match-mode overlap for the lenient rule)
.venv\Scripts\python.exe src\evaluate.py annotations\test_video_task_annotations.csv `
    outputs\detected_events.csv --tolerance 0.5
```

`--threshold`, `--velocity-threshold`, `--enter-frames`, `--exit-frames`, `--head-up` and
`--head-radius` control the head zone, gate and state machine; the defaults are
`1.0`, `1.5`, `3`, `10`, `0.6`, `0.6`. `--signal-csv` caches the raw per-frame geometry
(head centre and scale, wrist and fingertip positions), so retuning or trying another head
shape never needs the (expensive) landmark extraction again:

```powershell
# gate / state-machine settings for the current head shape
.venv\Scripts\python.exe src\tune_threshold.py outputs\distance_signal.csv `
    annotations\test_video_task_annotations.csv
# compare head shapes (the Pass 6 experiment) incl. leave-one-out
.venv\Scripts\python.exe src\geometry_experiment.py outputs\distance_signal.csv `
    annotations\test_video_task_annotations.csv
# re-run event extraction from the cache
.venv\Scripts\python.exe src\head_touch_detector.py data\test_video_task.mp4 `
    outputs\detected_events.csv --from-signal outputs\distance_signal.csv
```

### Live mode (webcam)

```powershell
.venv\Scripts\python.exe src\live_demo.py                              # default webcam; q or Esc quits
.venv\Scripts\python.exe src\live_demo.py --record outputs\demo.mp4    # also save the annotated feed
# a video file works as a stand-in for a camera; --start / --duration pick an excerpt
.venv\Scripts\python.exe src\live_demo.py --source data\test_video_task.mp4 --start 38 --duration 32 `
    --no-window --record outputs\clip.mp4
```

It shows the head zone, the hand landmarks with each hand's distance and speed, a green
**TOUCH** banner while a touch is confirmed, and a running log, and prints `TOUCH START` /
`TOUCH END` lines as they happen.

**Same code as the batch detector.** Per-frame analysis (`analyze_frame`) and the touch
state machine (`TouchDetector`, one `HandTouchTracker` per hand) are shared; the batch
`extract_events` is that same code run over a finished recording. `tests/` checks this
(`python -m unittest discover -s tests`, 12 tests): the frame-by-frame detector gives
exactly the same events as the original whole-recording implementation across 48
combinations of settings on the real signal, and synthetic cases cover the behaviours that
matter live — a fast sweep through the zone is ignored, a hand held still is one touch, a
brief dip is ignored, a hand lost for a few frames mid-touch does not split the touch, and
a touch still open when the stream stops is closed.

**End-to-end check.** Streaming the whole recording (9,349 frames) through `live_demo.py` at
native resolution gave exactly the same 6 events, with identical start, contact and end
times, as the batch run. Touch alerts arrived 0.07 s (2 frames) after the touch began.

**What to expect live, and what has not been validated**
- **Latency.** A `TOUCH START` alert appears after `--enter-frames` (3) qualifying frames.
  The end, and therefore the final contact time, is only known after `--exit-frames` (10)
  frames that fail the gate.
- **A hand that leaves the frame.** Frames where a hand is not detected do not end a touch
  (so occlusion by the head does not split it). Live, that would leave a touch open forever
  after the hand is lowered out of view, so `--lost-timeout` (default 1 s) closes a touch
  whose hand has not been seen for that long, at the last time it was seen. The batch
  path does not use this; it is off there.
- **Throughput.** On the development CPU the two MediaPipe models take about 60 ms per
  frame (hand 37 ms, pose 22 ms); the whole loop ran at 10 to 17 fps across runs. The
  enter/exit counts are in *frames* and were chosen on a 30 fps recording, so at 10 fps
  they span 3x more real time (about 0.3 s to confirm a touch, 1 s to end one). Speed is computed from real
  elapsed time, so the speed gate does not depend on frame rate, but the debounce does.
  `live_demo.py` prints a warning when it runs below 20 fps.
- **Resizing.** Frames are resized to 640 px wide by default for speed; landmarks shift
  slightly, so events differ a little from the native-resolution batch run (first touch:
  batch start 00:41.90 / contact 00:41.97, live at 640 px start 00:41.87 / contact 00:42.27).
- **Run on a webcam, but accuracy not measured live.** The author ran it on a laptop
  webcam and it worked (overlay and touch alerts behaved as expected), but that was an
  informal check. All accuracy numbers in this README come from one recorded video;
  other cameras, framing, lighting, mirrored views (which change MediaPipe's left/right
  labels) and low frame rates have not been measured.

### Demo videos

Two short excerpts of the test recording, run through the live pipeline
(`src/live_demo.py`) and saved with its overlay: the cyan circle is the head zone, the dots
are the hand landmarks, the banner turns green while a touch is confirmed, and the log lists
the start/end alerts. `t =` is the time in the original recording.

| Clip | Shows |
|---|---|
| [demo_1_two_touches_and_a_false_alarm.mp4](demo/demo_1_two_touches_and_a_false_alarm.mp4) (00:38–01:10) | Touch 1 (00:41.9) and touch 2 (01:04.6) detected, plus a **false alarm at 00:57.5** where both hands are clasped in front of the face. |
| [demo_2_three_touches_incl_quick_tap.mp4](demo/demo_2_three_touches_incl_quick_tap.mp4) (02:24–03:22) | Touches 3 (02:27.5), 4 (02:34.9) and the quick tap at 03:18.2 detected; open hands raised around 03:14 are correctly ignored. |

These are **a recording streamed through the live code, not a live camera**, processed at
640 px wide, so times differ slightly from the native-resolution results below. The excerpts
were **chosen** to show successes and a known failure, not sampled at random. To regenerate:

```powershell
.venv\Scripts\python.exe src\live_demo.py --source data\test_video_task.mp4 --start 38 --duration 32 `
    --no-window --record outputs\demo_raw_1.mp4
pip install imageio-ffmpeg    # dev-only, for the H.264 re-encode
.venv\Scripts\python.exe src\reencode_h264.py outputs\demo_raw_1.mp4 demo\demo_1.mp4
```

### Results

Run against `data/test_video_task.mp4` (5:12, 9,349 frames).

**Ground truth: 5 annotated touches, built up in three steps.**
1. The original annotation had 3 touches. **The gate settings of the first two passes were
   tuned on these 3 only.**
2. A 4th (right hand flat on the forehead, 02:34.03 / 02:34.90 / 02:36.10) was found during
   the false-positive review below and added.
3. A 5th (right hand, a quick touch, 03:17.70 / 03:18.13 / 03:18.60) was noticed afterwards
   and added.

Two of the five touches were therefore added after the detector's output had been seen;
see the limitations. **Pass 5** is the original head model with speed limit 0.5; **Pass 6**
is the raised, smaller head zone with speed limit 1.5 (the current defaults).

| 5 annotated touches | Pass 5, strict | Pass 5, overlap | **Pass 6, strict** | **Pass 6, overlap** |
|---|---|---|---|---|
| Detected events | 11 | 11 | 6 | 6 |
| True positives | 3 | 4 | 3 | 5 |
| False negatives (missed) | 2 | 1 | 2 | 0 |
| False positives (extra) | 8 | 7 | 3 | 1 |
| **Precision** | 0.27 | 0.36 | **0.50** | **0.83** |
| **Recall** | 0.60 | 0.80 | **0.60** | **1.00** |
| F1 | 0.37 | 0.50 | 0.55 | 0.91 |
| Mean timing error on matches | 0.26 s | 0.42 s | 0.17 s | 0.47 s |

Under the strict rule Pass 6 still misses touches 2 and 4, but they are *detected*: the
detected contact time (the closest frame) is 0.76 s and 1.07 s after the annotated
landing, outside ±0.5 s. Only the overlap rule counts them. Detected lag on the five
touches is +0.20, +0.76, +0.03, +1.07 and +0.27 s.

### Pass 6: a better head shape (in-sample; leave-one-out only partly agrees)

**Why.** Reviewing the Pass 5 false positives (next section) showed the same failure over
and over: an open hand raised beside the face fell inside the head circle. The circle was
centred at eye level with radius = the full head width, so it was too large and too low.
It also missed the quick tap at 03:18 because the 0.5 speed limit rejected it, and
loosening the limit alone brought in ~11 more false positives.

**The hypothesis itself came from looking at this video's frames** (the review below),
so it is not independent of the data it was then tested on.

**Protocol, fixed before running:** raise ∈ {0, .2, .4, .6}, radius ∈ {.5 … 1.0}, speed
limit ∈ {0.5, 1.0, 1.5, none}, other settings unchanged, selected by F1 under the overlap
rule ([src/geometry_experiment.py](src/geometry_experiment.py); full output in
[results/geometry_experiment.txt](results/geometry_experiment.txt)). The original model
reproduces the earlier distances to within 0.0002.

**In-sample result.** Best: raise 0.6, radius 0.6, speed limit 1.5 — all 5 touches
found, 1 extra detection (down from 7), the quick tap now detected, the raised-hand
detections gone.
- The direction is not a single spike: with speed limit 0.5, raise 0.6 with radius
  0.6–0.9, and raise 0.4 with radius 0.8–0.9, keep 4 touches with ≤ 1 extra detection
  (original model: 7). Too small a zone loses real touches (raise 0.6, radius 0.5 keeps 3).
- The winner is on the **edge of the tested grid** (0.6 is the largest raise tried); a
  better value may lie beyond it, which was deliberately not explored.

**Leave-one-out (the more honest estimate): 3 of 5 held-out touches found**, against 4 of 5
for the old fixed defaults. Holding out touch 1 chose radius 0.5, which loses touch 1;
holding out touch 5 chose the tight speed limit, since no other touch is a quick tap. So
the evidence supports the **direction** (a smaller, higher zone removes most false
positives), but the **specific values were fitted to these 5 touches in one video**, and
catching the quick tap depends on that touch itself. Treat 0.83 precision as in-sample.

Because the zone sits over the top of the head, contacts with the lower face (chin, jaw,
eye) are no longer detected. That matches the ground truth, which counts head/forehead
contact only, but it is a change in what the detector responds to, not only an accuracy gain.

### What still goes wrong

The one remaining extra detection (00:57.60) is both hands clasped **in front of** the
face: the fingertips overlap the zone in the image but the hands are not on the head. The
detector is 2D, so it cannot tell the two apart (`results/frames/FP_det02_00m57.60s.jpg`).

### Review of the Pass 5 false positives

Each of the 8 extra detections of the original model (Pass 5, strict rule) was inspected by
viewing the frame at its contact time (`results/frames_pass5/`, generated by
`src/visualize_events.py --head-up 0 --head-radius 1`). This is one frame per event,
reviewed by eye, not a re-annotation. The review surfaced the missed forehead touch
(detection #5), which was then added to the ground truth.

| Det | Time | Hand | What the frame shows | Verdict |
|---|---|---|---|---|
| 2 | 00:47.77 | both | Both open hands raised beside the face, thumbs inside the circle | False positive |
| 5 | 02:35.80 | right | Right hand flat on the forehead | **Real head touch; now annotated as GT#4** |
| 6 | 03:14.00 | both | Both open hands raised beside the face | False positive |
| 7 | 03:14.80 | both | Both open hands raised beside the face | False positive |
| 8 | 03:16.60 | both | Both open hands raised beside the face | False positive |
| 9 | 03:20.13 | right | Open hand raised at the side of the face (waving), thumb at the circle edge | False positive |
| 10 | 03:47.80 | left | Hand on the chin / jaw, covering the lower face | Face contact, not annotated |
| 11 | 05:06.47 | right | Hand over the eye (rubbing / covering the eye) | Face contact, not annotated |

Five of the eight are open hands raised beside the face, one is the real forehead touch,
and two are lower-face/eye contacts that the ground truth deliberately does not count
(counting them would have made the Pass 5 result 6 of 11 correct — indicative only). The
5th annotated touch (03:18) had no detection in Pass 5, shown in
`results/frames_pass5/FN_gt05_03m18.13s.jpg`. `results/frames/` holds the current (Pass 6)
frames: 5 `TP_*` and 1 `FP_*`.

### Worked example

Ground-truth touch #3 is annotated with contact at **02:27.93**. The detector reports
start **02:27.47**, contact **02:27.90**, min normalized distance **0.087**, i.e. a
fingertip deep inside the head zone while the wrist was moving slower than 1.5
head-widths/sec for 3+ consecutive frames. Timing error: **0.03 s**.

## Limitations and honest notes

- **What this is.** A rule-based detector on top of *pretrained* MediaPipe landmark
  models. Nothing here is trained on this project's data; the annotations are used only to
  evaluate the detector and to choose its settings (head shape, speed limit, distance
  threshold, enter/exit frame counts).
- **Results are in-sample.** The head shape and speed limit were chosen against all 5
  touches from one person, one camera and one recording (the earlier gate settings against
  the original 3). Leave-one-out found 3 of 5 held-out touches. The numbers describe that
  recording, not new footage. With 5 events, one event more or less moves recall by 20
  points.
- **The ground truth changed after the detector ran.** Two touches were added afterwards:
  the 4th was found through the false-positive review, the 5th was noticed later. These
  are corrections of annotation errors, not tuning, but the ground truth is not fully
  independent of the detector's output. The lenient overlap rule was likewise added after
  seeing results.
- **The tested head shapes are a small, fixed family** and the chosen one is at its edge;
  larger raises and non-circular (elliptical) zones were not tried.
- **Detected `contact_time` lags the landing.** It is the frame where the hand is closest,
  while the annotations mark where it lands (+0.20, +0.76, +0.03, +1.07, +0.27 s in Pass
  6). This is why the strict rule still misses two detected touches.
- **Distance + speed can't see contact.** The zone is 2D, so a hand held *in front of* the
  head toward the camera looks identical to one touching it (the remaining false positive).
  The zone also covers only the top of the head, so lower-face contacts are not detected.
- **Hand identity is just the MediaPipe label.** In 58 (frame, hand) pairs both hands get
  the same label, which merges two hands into one track for speed and event extraction.
- **Pose vs. Face Mesh was reasoned, not tested.** Pose landmarks were chosen for the head
  because face tracking is expected to degrade when a hand covers the face. The two were
  not compared empirically on this video.
- **Coverage gaps.** Only 3,407 (frame, hand) rows exist for 9,349 frames: when no hand
  or no usable head estimate is found the frame produces no row, so touches during
  detection dropouts can be missed.
- **Hand labels happened to match.** MediaPipe's left/right labels agreed with the
  annotations on every matched touch for this video, so no mirroring correction was needed;
  this may not hold for other cameras (e.g. mirrored webcams). Matching therefore ignores
  the label and reports it as a diagnostic.
- **Live accuracy is unmeasured.** Live mode was checked by streaming a recording through
  the live loop (identical events to the batch run) and informally on a laptop webcam, but
  no accuracy was measured on a live feed (see Live mode).
- **Toward production.** Validate on multiple people/cameras/lighting first. If precision
  is still inadequate, train a small classifier over the *existing* features (distance,
  speed, dwell time) rather than end-to-end video models; both need labeled footage
  beyond this one recording.

## Reproducing

The source video is a personal recording and is **not in the repo** (it is 345 MB, over
GitHub's 100 MB file limit; it can be shared separately on request), and the model
bundles are not in the repo either. The committed
`results/` folder holds the detected events, the per-frame signal (with the raw geometry)
and annotated frames from the run reported here, and `annotations/` holds the ground truth,
so the evaluation, gate tuning and the Pass 6 head-shape experiment can all be re-run
from `results/distance_signal.csv` without the video. Re-running the detection itself needs a video: put
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
- **A head zone, not the whole head.** The detector asks whether a hand landmark is inside
  a circle over the top of the head (raised 0.6, radius 0.6 head-widths). It was chosen by a
  small fixed experiment (Pass 6) after the original, larger eye-level circle produced
  false positives; the evidence and its limits are in the Results section.
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
