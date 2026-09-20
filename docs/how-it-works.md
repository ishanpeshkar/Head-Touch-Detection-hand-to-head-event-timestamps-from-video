# How the head-touch detector works

The system takes a video and returns timestamps of hand-to-head touches. It is a
rule-based detector built on top of pretrained landmark models; nothing in it is trained
on this project's data. Each frame goes through four stages, and the per-frame results
are then turned into discrete events.

```
video frame ──► 1. landmarks ──► 2. geometry ──► 3. speed gate ──► 4. event state machine ──► events CSV
              (MediaPipe pose      (head circle,    (wrist speed,      (debounce, start /        (start, contact,
               + hand models)       normalized       head-widths/sec)   contact / end)             end, hand)
                                    distance)
```

## 1. Landmarks (pretrained)

MediaPipe's **Pose Landmarker** provides the head points (nose, eyes, ears). Its
**Hand Landmarker** provides 21 points per hand plus a left/right label. Both are generic
models published by Google and used as-is, through the MediaPipe Tasks API in `VIDEO`
mode (which uses tracking between frames).

Pose landmarks are used for the head rather than Face Mesh because the moment of contact
is when the face is partly occluded by the hand, and face tracking is expected to degrade
under occlusion. **This is a reasoned choice; the two were not compared on this video.**

## 2. Geometry: the head as a circle

- **Centre:** the average of the visible head points (nose, eyes, ears; each must have
  visibility >= 0.5, and at least two must be visible).
- **Radius (`head_scale`):** the ear-to-ear distance. If the ears are not both visible,
  it falls back to 2.6 x the eye-to-eye distance, then to 0.55 x the shoulder width.
- **Distance:** for each detected hand, take the closest of six landmarks (the wrist and
  the five fingertips) to the head centre, and divide by `head_scale`. A value of 1.0 is
  exactly the edge of the circle; below 1.0 means that landmark is inside it.
- Dividing by head size makes the value comparable whether the person is near or far from
  the camera.

## 3. Speed gate

A frame counts as "touching" only if the hand is inside the circle **and** slow. A real
touch stops the hand; a gesture passes through the circle quickly.

- Speed is the **wrist's** pixel speed, divided by `head_scale`, so it is in
  **head-widths per second**, smoothed over 3 frames.
- The wrist is used rather than the closest fingertip because the "closest" landmark can
  switch fingers from frame to frame and would fake motion on a hand that is still.
- If the hand was not detected for more than 0.25 s, speed is treated as unknown (not
  slow).
- Threshold: 0.5 head-widths/sec.

## 4. Event state machine

Turns the per-frame flags into events, per hand label:

- **Start:** 3 consecutive detected frames that pass the gate. The reported start time is
  back-dated to the first of those frames.
- **End:** 10 consecutive frames that fail the gate.
- **Detection dropouts:** frames where the hand is not detected do not count toward the
  end, so a hand that is briefly hidden by the head or its own occlusion mid-touch keeps
  the touch open.
- **Contact time:** the frame with the smallest distance within 1.5 s of the start. The
  window is bounded because a hand lingering near the head can keep one event open for
  many seconds, and an unbounded minimum could pick an unrelated later dip. Note that this
  frame is chosen by distance only; the hand can still be moving at that frame.

## Worked example

Ground-truth touch #3 is annotated with contact at **02:27.93**. The detector reports
start **02:27.63** and contact **02:27.90**, with a minimum normalized distance of **0.571**
(an index fingertip well inside the circle). Timing error: **0.03 s**.

![True positive: det#4, matches ground-truth touch #3](../results/frames/TP_det04_02m27.90s.jpg)

The drawing shows the head circle (cyan; its edge is distance 1.0), the head keypoints
(white), the wrist (yellow) and fingertips (green) of each hand, and the closest hand
landmark joined to the head centre (red).

## Evaluation and tuning

- **`evaluate.py`** matches detected events to annotated events one-to-one by nearest
  `contact_time` within +/-0.5 s, and reports precision, recall, F1 and timing error. The
  hand label is reported but does not decide a match.
- **`tune_threshold.py`** grid-searches four settings (distance threshold, speed threshold,
  enter frames, exit frames) against the annotations, using the cached per-frame signal so
  the slow landmark pass is never repeated.

Result on the recording, 3 annotated touches: **3 of 3 found, 8 extra detections,
precision 0.27, recall 1.00**, contact times within 0.03-0.46 s of the annotations.
(Distance-only, without the speed gate: 21 extra detections, precision 0.12.)

## Files

| File | Role |
|---|---|
| `src/inspect_video.py`, `src/annotate.py` | Video metadata; interactive ground-truth annotation tool |
| `src/head_touch_detector.py` | The core: landmarks, distance, speed, event extraction |
| `src/evaluate.py`, `src/tune_threshold.py` | Scoring and parameter search |
| `src/visualize_events.py` | Draws the detector's view on each detected contact frame |
| `src/timeutils.py`, `src/download_models.py` | Shared time helpers; model download |

## What the false positives look like

All 8 extra detections were reviewed by looking at the frame at each one (see the README
for the full table and `results/frames/`). Five are **open hands raised beside the face**;
three are **genuine hand-on-face contacts that were not in the ground truth**. The
frames also show that the head circle is larger than the head and centred at eye level,
which is the main cause of the false positives and also why some real touches at the
hairline are detected via the wrist rather than the fingertips.

![False positive: open hands raised beside the face](../results/frames/FP_det02_00m47.77s.jpg)

![Detected but not annotated: hand on the forehead](../results/frames/FP_det05_02m35.80s.jpg)

## Design rationale and caveats

- **Pose over Face Mesh** is a hypothesis about occlusion robustness, not a measured
  result.
- **The results are in-sample.** The four settings were tuned on 3 events from one person
  and one camera, so 0.27 precision says little about new footage.
- **Recall was favoured over precision** on purpose: missing a real touch is worse than
  flagging an extra candidate a person can dismiss.
- **Rule-based, not trained.** The natural next step is a small classifier over the
  existing features (distance, speed, dwell time), but only once footage from several
  people and cameras is available. Better head geometry (a smaller circle centred higher,
  or an ellipse) is a cheaper first step.
