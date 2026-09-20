"""Phase 3+4: detect hand-to-head contact events in a video.

Method (see README for the full write-up):

1. Per frame, run MediaPipe's Pose Landmarker (for head keypoints: nose,
   eyes, ears) and Hand Landmarker (for fingertip keypoints), both via the
   Tasks API in VIDEO mode.

   Pose is used for the head instead of Face Mesh on purpose: the moment a
   hand touches the head is exactly when the face becomes partially
   occluded, and the pose model degrades far more gracefully under partial
   occlusion than a dense face mesh does.

2. Approximate the head as a circle: its center is the (visibility-
   weighted) centroid of the visible head keypoints, and its radius
   ("head_scale") is a stable head-width estimate, preferring, in order:
   ear-to-ear distance, else 2.6x eye-to-eye distance, else 0.55x
   shoulder-to-shoulder distance. All three are well-known anthropometric
   proxies for head width; the fallback chain keeps the estimate available
   even when part of the face/head is turned away or occluded.

3. For each detected hand, take the minimum pixel distance from its
   fingertip + wrist landmarks to the head center, and normalize it by
   head_scale. A normalized distance <= ~1.0 means a hand landmark has
   entered the head disk.

4. Gate on speed as well as distance: a frame only counts as touching if
   the wrist is also moving slowly (a real touch decelerates to a near
   standstill at the head; a gesture passes through at speed). Velocity
   is wrist speed in head-widths/sec, see `add_velocities`.

5. Turn that per-frame signal into discrete events with a small debounce
   state machine (see `extract_events`): a few consecutive frames must
   cross the threshold before a touch is confirmed as starting or ending,
   so single-frame landmark jitter doesn't fabricate events. A frame where
   the hand briefly isn't detected *while already touching* is treated as
   "still touching" rather than an automatic end, since self-occlusion
   from the hand covering part of the head is expected right when contact
   happens.

The event logic is incremental (HandTouchTracker / TouchDetector), so the same code
runs on a finished recording here and frame by frame in live_demo.py.

This is deliberately a hand-crafted geometric baseline, not a trained
classifier -- consistent with the project's Phase 1/2 "no ML training
yet" approach.

Usage:
    python src/head_touch_detector.py data/test_video_task.mp4 \\
        outputs/detected_events.csv \\
        --signal-csv outputs/distance_signal.csv
"""

import argparse
import csv
import os
from collections import deque

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from timeutils import format_timestamp

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
HAND_MODEL_PATH = os.path.join(MODELS_DIR, "hand_landmarker.task")
POSE_MODEL_PATH = os.path.join(MODELS_DIR, "pose_landmarker_lite.task")

PL = vision.PoseLandmark
HEAD_LANDMARK_IDS = [PL.NOSE, PL.LEFT_EYE, PL.RIGHT_EYE, PL.LEFT_EAR, PL.RIGHT_EAR]
VISIBILITY_THRESHOLD = 0.5

# Hand landmark indices that can plausibly make contact with the head.
HAND_CONTACT_LANDMARK_IDS = [0, 4, 8, 12, 16, 20]  # wrist + 5 fingertips

EVENTS_CSV_HEADER = ["event_id", "start_time", "contact_time", "end_time", "hand", "notes"]
# Raw geometry is stored alongside the distance so head-shape variants (see
# apply_head_geometry) can be tested offline without re-running the landmarkers.
CONTACT_COLUMNS = [f"c{i}_{axis}" for i in HAND_CONTACT_LANDMARK_IDS for axis in ("x", "y")]
SIGNAL_CSV_HEADER = ["frame", "time_sec", "hand", "normalized_distance", "head_scale_px", "head_source",
                     "wrist_x", "wrist_y", "head_cx", "head_cy"] + CONTACT_COLUMNS

WRIST_ID = 0
# Velocity is only trusted across short gaps in hand detection; a longer gap
# means the hand was lost and re-found, so the "speed" between the two
# observations isn't meaningful and is treated as unknown (i.e. not slow).
MAX_VELOCITY_GAP_SEC = 0.25
VELOCITY_SMOOTHING_ROWS = 3


def _dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _to_px(landmark, width, height):
    return (landmark.x * width, landmark.y * height)


def head_center_and_scale(pose_landmarks, width, height):
    """Return ((cx, cy), scale_px, source) for one detected pose, or None
    if too few head keypoints are visible to make an estimate."""
    visible = {}
    for lm_id in HEAD_LANDMARK_IDS:
        lm = pose_landmarks[lm_id]
        if lm.visibility >= VISIBILITY_THRESHOLD:
            visible[lm_id] = _to_px(lm, width, height)

    if len(visible) < 2:
        return None

    cx = sum(p[0] for p in visible.values()) / len(visible)
    cy = sum(p[1] for p in visible.values()) / len(visible)
    center = (cx, cy)

    if PL.LEFT_EAR in visible and PL.RIGHT_EAR in visible:
        scale = _dist(visible[PL.LEFT_EAR], visible[PL.RIGHT_EAR])
        source = "ear"
    elif PL.LEFT_EYE in visible and PL.RIGHT_EYE in visible:
        scale = _dist(visible[PL.LEFT_EYE], visible[PL.RIGHT_EYE]) * 2.6
        source = "eye"
    else:
        left_sh, right_sh = pose_landmarks[PL.LEFT_SHOULDER], pose_landmarks[PL.RIGHT_SHOULDER]
        if left_sh.visibility >= VISIBILITY_THRESHOLD and right_sh.visibility >= VISIBILITY_THRESHOLD:
            scale = _dist(_to_px(left_sh, width, height), _to_px(right_sh, width, height)) * 0.55
            source = "shoulder"
        else:
            return None

    if scale <= 1e-3:
        return None
    return center, scale, source


def create_landmarkers(max_hands: int = 2):
    """Return (hand_landmarker, pose_landmarker), both in VIDEO mode."""
    hand_landmarker = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )
    pose_landmarker = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )
    return hand_landmarker, pose_landmarker


def analyze_frame(pose_result, hand_result, width, height, frame_idx, time_sec):
    """Turn one frame's MediaPipe results into signal rows.

    Returns (rows, head): one row dict per detected hand (empty when there is no
    usable head estimate or no hand) and head = (center, scale, source) or None.
    Shared by the batch pass (compute_distance_signal) and the live demo, so both
    compute identical rows from the same landmarks.
    """
    rows, head = [], None
    if pose_result.pose_landmarks:
        head = head_center_and_scale(pose_result.pose_landmarks[0], width, height)
    if head is not None and hand_result.hand_landmarks:
        center, scale, source = head
        for hand_landmarks, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
            hand_label = handedness[0].category_name  # 'Left' / 'Right'
            contact_px = {i: _to_px(hand_landmarks[i], width, height)
                          for i in HAND_CONTACT_LANDMARK_IDS}
            min_px_dist = min(_dist(p, center) for p in contact_px.values())
            wrist_x, wrist_y = _to_px(hand_landmarks[WRIST_ID], width, height)
            rows.append({
                "frame": frame_idx,
                "time_sec": time_sec,
                "hand": hand_label,
                "normalized_distance": min_px_dist / scale,
                "head_scale_px": scale,
                "head_source": source,
                "wrist_x": wrist_x,
                "wrist_y": wrist_y,
                "head_cx": center[0],
                "head_cy": center[1],
                **{f"c{i}_{axis}": contact_px[i][k]
                   for i in HAND_CONTACT_LANDMARK_IDS for k, axis in enumerate(("x", "y"))},
            })
    return rows, head


def compute_distance_signal(video_path: str, max_hands: int = 2):
    """Run pose + hand landmarkers over every frame and yield one dict per
    (frame, detected hand): {frame, time_sec, hand, normalized_distance,
    head_scale_px, head_source, ...raw geometry}. Frames with no usable head
    estimate or no detected hand are skipped (no row emitted)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    hand_landmarker, pose_landmarker = create_landmarkers(max_hands)

    frame_idx = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            timestamp_ms = int(round(frame_idx / fps * 1000))
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)
            hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

            if frame_idx % 300 == 0:
                print(f"  frame {frame_idx}/{frame_count} ({frame_idx / fps:.1f}s)")

            rows, _ = analyze_frame(pose_result, hand_result, width, height, frame_idx, frame_idx / fps)
            yield from rows
            frame_idx += 1
    finally:
        cap.release()
        hand_landmarker.close()
        pose_landmarker.close()


class VelocityTracker:
    """Wrist speed for ONE hand, in head-widths per second, fed one row at a time.

    The wrist is used as the anchor rather than the closest-of-several contact
    landmarks used for distance, because that "closest" landmark can flip between
    fingertips frame to frame and fake motion on a hand that is actually still.
    Step speeds are smoothed over a short trailing window since single-frame
    landmark jitter is comparable to real slow motion. A row with no usable
    predecessor (first row, or a gap longer than MAX_VELOCITY_GAP_SEC) gets `inf`,
    i.e. never counts as slow.
    """

    def __init__(self):
        self._prev = None
        self._steps = deque(maxlen=VELOCITY_SMOOTHING_ROWS)

    def update(self, row) -> float:
        speed = float("inf")
        prev = self._prev
        if prev is not None:
            dt = row["time_sec"] - prev["time_sec"]
            if 0 < dt <= MAX_VELOCITY_GAP_SEC:
                px = _dist((row["wrist_x"], row["wrist_y"]), (prev["wrist_x"], prev["wrist_y"]))
                speed = px / dt / row["head_scale_px"]
        self._prev = row
        self._steps.append(speed)
        return sum(self._steps) / len(self._steps)


def add_velocities(rows) -> None:
    """Annotate time-sorted rows of ONE hand with a `velocity` key
    (see VelocityTracker)."""
    tracker = VelocityTracker()
    for row in rows:
        row["velocity"] = tracker.update(row)


class HandTouchTracker:
    """Frame-by-frame touch state machine for ONE hand label.

    Feed rows in time order with update(); it returns messages as they become
    known, so it works identically on a finished recording and on a live feed:

      {"type": "start", ...}  a touch is confirmed (after `enter_frames` frames that
                              pass the gate); start_time is back-dated to the first
                              of those frames, detected_time is when it was confirmed
      {"type": "end", "event": {...}}  the touch is over (after `exit_frames` frames
                              that fail the gate); the event carries start_time,
                              contact_time, end_time and min_normalized_distance

    `contact_window_seconds` bounds how far past `start_time` we look for the
    "contact" frame (the local minimum of the distance signal). This matters
    because the hand can linger near the threshold for a long stretch (fidgeting,
    resting near the face after a real touch) without the exit debounce firing,
    which keeps a single event "open" for much longer than the actual touch lasted.
    Searching the *whole* open stretch for a global minimum can then pick a later,
    unrelated dip as "contact". Ground-truth touches in this project are
    consistently under ~1.5s from start to end, so that's the default search
    window; `end_time` (and the exit debounce that produces it) is unaffected and
    can still extend well past it.

    `velocity_threshold` (head-widths/sec, None = off) adds a second gate: a frame
    only counts as "touching" if the hand is close AND not moving too fast. A hand
    resting on the head is slow, while a gesture sweeping through the zone is fast.
    Because the gate feeds the same enter/exit debounce, `enter_frames` doubles as
    the dwell-time requirement.

    Frames where the hand is not detected simply never reach update(), so a hand
    briefly hidden mid-touch neither confirms nor ends a touch. On a recording that is
    harmless (the next detection closes the touch), but on a live feed a hand that
    leaves the frame would leave the touch open forever. `lost_timeout_seconds`
    (None = off, the batch default) fixes that: call tick(now) every frame, and a touch
    whose hand has not been seen for that long is closed at the last time it was seen.
    """

    def __init__(self, hand, threshold, enter_frames, exit_frames,
                 contact_window_seconds=1.5, velocity_threshold=None, lost_timeout_seconds=None):
        self.hand = hand
        self.threshold = threshold
        self.enter_frames = enter_frames
        self.exit_frames = exit_frames
        self.contact_window_seconds = contact_window_seconds
        self.velocity_threshold = velocity_threshold
        self.lost_timeout_seconds = lost_timeout_seconds
        self._velocity = VelocityTracker() if velocity_threshold is not None else None
        # Enough history to back-date the start (enter_frames rows) and the end
        # (exit_frames + 1 rows) without keeping the whole stream.
        self._history = deque(maxlen=max(enter_frames, exit_frames + 1))
        self.state = "idle"
        self._below = 0
        self._above = 0
        self._start_row = None
        self._best_row = None  # min normalized_distance row within the contact window

    @property
    def touching(self) -> bool:
        return self.state == "touching"

    def _event(self, end_time):
        best = self._best_row
        return {
            "hand": self.hand,
            "start_time": self._start_row["time_sec"],
            "contact_time": best["time_sec"],
            "end_time": end_time,
            "min_normalized_distance": best["normalized_distance"],
        }

    def update(self, row):
        messages = []
        if self._velocity is not None:
            row["velocity"] = self._velocity.update(row)
        self._history.append(row)

        below = row["normalized_distance"] <= self.threshold
        if self._velocity is not None:
            below = below and row["velocity"] <= self.velocity_threshold

        if self.state == "idle":
            self._below = self._below + 1 if below else 0
            if self._below >= self.enter_frames:
                # Back-date the start to where the run of "below" frames began.
                self._start_row = self._history[-self.enter_frames]
                self._best_row = row
                self.state = "touching"
                self._above = 0
                messages.append({"type": "start", "hand": self.hand,
                                 "start_time": self._start_row["time_sec"],
                                 "detected_time": row["time_sec"]})
        else:  # touching
            if (row["time_sec"] - self._start_row["time_sec"] <= self.contact_window_seconds
                    and row["normalized_distance"] < self._best_row["normalized_distance"]):
                self._best_row = row
            self._above = self._above + 1 if not below else 0
            if self._above >= self.exit_frames:
                h = self._history
                end_row = h[-(self.exit_frames + 1)] if len(h) > self.exit_frames else h[0]
                # Back-dating the end by exit_frames can land before the contact
                # frame on short events; an event can't end before it touches.
                end_time = max(end_row["time_sec"], self._best_row["time_sec"])
                messages.append({"type": "end", "event": self._event(end_time)})
                self.state = "idle"
                self._below = 0
                self._start_row = None
                self._best_row = None
        return messages

    def tick(self, now):
        """Advance the clock on a frame where this hand was not seen. Closes an open
        touch whose hand has been missing longer than `lost_timeout_seconds`."""
        if (self.state != "touching" or self.lost_timeout_seconds is None
                or now - self._history[-1]["time_sec"] <= self.lost_timeout_seconds):
            return []
        end_time = max(self._history[-1]["time_sec"], self._best_row["time_sec"])
        event = self._event(end_time)
        self.state = "idle"
        self._below = 0
        self._start_row = None
        self._best_row = None
        return [{"type": "end", "event": event, "reason": "hand lost"}]

    def flush(self):
        """Close a touch that is still open when the stream ends."""
        if self.state != "touching":
            return None
        event = self._event(self._history[-1]["time_sec"])
        self.state = "idle"
        self._start_row = None
        self._best_row = None
        return event


class TouchDetector:
    """One HandTouchTracker per hand label, fed the rows of each frame."""

    def __init__(self, threshold, enter_frames, exit_frames,
                 contact_window_seconds=1.5, velocity_threshold=None, lost_timeout_seconds=None):
        self._kwargs = dict(threshold=threshold, enter_frames=enter_frames, exit_frames=exit_frames,
                            contact_window_seconds=contact_window_seconds,
                            velocity_threshold=velocity_threshold,
                            lost_timeout_seconds=lost_timeout_seconds)
        self.trackers = {}

    def update(self, rows):
        messages = []
        for row in rows:
            tracker = self.trackers.get(row["hand"])
            if tracker is None:
                tracker = self.trackers[row["hand"]] = HandTouchTracker(row["hand"], **self._kwargs)
            messages.extend(tracker.update(row))
        return messages

    def tick(self, now):
        """Call once per frame after update(): closes touches whose hand has vanished."""
        messages = []
        for tracker in self.trackers.values():
            messages.extend(tracker.tick(now))
        return messages

    @property
    def touching_hands(self):
        return [hand for hand, t in self.trackers.items() if t.touching]

    def flush(self):
        return [e for e in (t.flush() for t in self.trackers.values()) if e is not None]


def extract_events(signal_rows, threshold: float, enter_frames: int, exit_frames: int,
                    contact_window_seconds: float = 1.5, velocity_threshold: float = None):
    """Batch entry point: run a whole recorded signal through the same
    frame-by-frame TouchDetector that the live demo uses, and return the events
    as a list of dicts with keys start_time, contact_time, end_time, hand (all in
    seconds) plus min_normalized_distance for diagnostics. See HandTouchTracker for
    the meaning of the parameters."""
    detector = TouchDetector(threshold, enter_frames, exit_frames,
                             contact_window_seconds, velocity_threshold)
    # Keep events grouped by hand in first-seen order, then sort by start time, so
    # ties between hands keep a stable, reproducible order.
    events_by_hand = {}
    for row in signal_rows:
        events_by_hand.setdefault(row["hand"], [])
    for row in sorted(signal_rows, key=lambda r: r["frame"]):
        for msg in detector.update([row]):
            if msg["type"] == "end":
                events_by_hand[msg["event"]["hand"]].append(msg["event"])
    for event in detector.flush():
        events_by_hand[event["hand"]].append(event)

    events = [e for hand_events in events_by_hand.values() for e in hand_events]
    events.sort(key=lambda e: e["start_time"])
    return events


def write_signal_csv(path: str, rows) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(SIGNAL_CSV_HEADER)
        for row in rows:
            writer.writerow([
                row["frame"], f"{row['time_sec']:.3f}", row["hand"],
                f"{row['normalized_distance']:.4f}", f"{row['head_scale_px']:.2f}", row["head_source"],
                f"{row['wrist_x']:.2f}", f"{row['wrist_y']:.2f}",
                f"{row['head_cx']:.2f}", f"{row['head_cy']:.2f}",
                *[f"{row[c]:.2f}" for c in CONTACT_COLUMNS],
            ])


def load_signal_csv(path: str):
    """Load a previously-written --signal-csv back into the row dicts
    extract_events expects, so thresholds/debounce can be retuned without
    re-running the (slow) landmark extraction. See tune_threshold.py."""
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append({
                "frame": int(row["frame"]),
                "time_sec": float(row["time_sec"]),
                "hand": row["hand"],
                "normalized_distance": float(row["normalized_distance"]),
                "head_scale_px": float(row["head_scale_px"]),
                "head_source": row["head_source"],
                "wrist_x": float(row["wrist_x"]),
                "wrist_y": float(row["wrist_y"]),
                # Raw geometry; absent in signal CSVs written before it was stored.
                **({"head_cx": float(row["head_cx"]), "head_cy": float(row["head_cy"]),
                    **{c: float(row[c]) for c in CONTACT_COLUMNS}}
                   if "head_cx" in row else {}),
            })
    return rows


def apply_head_geometry(rows, up: float = 0.0, radius: float = 1.0) -> None:
    """Recompute each row's normalized_distance for a different head shape.

    The head is a circle centred `up` x head_scale above the eye-level centre,
    with radius `radius` x head_scale. up=0, radius=1 is the original model
    (radius = full ear-to-ear width, centred at eye level) and reproduces the
    distances computed at extraction time. Needs the raw geometry columns.
    """
    for row in rows:
        cx, cy = row["head_cx"], row["head_cy"] - up * row["head_scale_px"]
        r = radius * row["head_scale_px"]
        row["normalized_distance"] = min(
            _dist((row[f"c{i}_x"], row[f"c{i}_y"]), (cx, cy)) for i in HAND_CONTACT_LANDMARK_IDS
        ) / r


def write_events_csv(path: str, events) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(EVENTS_CSV_HEADER)
        for i, e in enumerate(events, start=1):
            notes = f"auto-detected; min_normalized_distance={e['min_normalized_distance']:.3f}"
            writer.writerow([
                i, format_timestamp(e["start_time"]), format_timestamp(e["contact_time"]),
                format_timestamp(e["end_time"]), e["hand"].lower(), notes,
            ])


def main():
    parser = argparse.ArgumentParser(description="Detect hand-to-head contact events in a video.")
    parser.add_argument("video", help="Path to the input video file")
    parser.add_argument("events_csv", help="Path to write detected events CSV to")
    parser.add_argument("--signal-csv", default=None,
                         help="Optional path to also dump the raw per-frame distance signal")
    parser.add_argument("--from-signal", default=None,
                         help="Skip landmark extraction and re-run event extraction from a "
                              "previously-written --signal-csv (fast; for tuning thresholds)")
    parser.add_argument("--threshold", type=float, default=1.0,
                         help="Normalized distance below which a hand counts as touching the head")
    parser.add_argument("--enter-frames", type=int, default=3,
                         help="Consecutive below-threshold frames required to confirm touch start")
    parser.add_argument("--exit-frames", type=int, default=10,
                         help="Consecutive above-threshold frames required to confirm touch end")
    parser.add_argument("--contact-window", type=float, default=1.5,
                         help="Seconds after touch start to search for the contact (min-distance) frame")
    parser.add_argument("--velocity-threshold", type=float, default=1.5,
                         help="Max wrist speed (head-widths/sec) for a frame to count as touching; "
                              "omit to disable the velocity gate")
    parser.add_argument("--head-up", type=float, default=0.6,
                         help="Raise the head circle's centre by this many head-widths above the "
                              "eye-level centre (0 = original model)")
    parser.add_argument("--head-radius", type=float, default=0.6,
                         help="Head circle radius in head-widths (1.0 = original model)")
    parser.add_argument("--max-hands", type=int, default=2)
    args = parser.parse_args()

    if args.from_signal:
        print(f"Loading cached signal from {args.from_signal} ...")
        signal_rows = load_signal_csv(args.from_signal)
    else:
        print(f"Processing {args.video} ...")
        signal_rows = list(compute_distance_signal(args.video, max_hands=args.max_hands))
    print(f"Collected {len(signal_rows)} (frame, hand) rows.")

    if args.signal_csv and not args.from_signal:
        write_signal_csv(args.signal_csv, signal_rows)
        print(f"Wrote raw distance signal to {args.signal_csv}")

    # Applied after the signal is written, so a cached signal always holds the
    # original-geometry distances plus the raw geometry needed to change the shape.
    if (args.head_up, args.head_radius) != (0.0, 1.0):
        try:
            apply_head_geometry(signal_rows, args.head_up, args.head_radius)
        except KeyError:
            raise SystemExit("This signal CSV predates stored geometry; re-run extraction, or pass "
                             "--head-up 0 --head-radius 1 to use the original head model.")

    events = extract_events(signal_rows, args.threshold, args.enter_frames, args.exit_frames,
                             contact_window_seconds=args.contact_window,
                             velocity_threshold=args.velocity_threshold)
    write_events_csv(args.events_csv, events)
    print(f"Detected {len(events)} event(s), written to {args.events_csv}")
    for e in events:
        print(f"  {e['hand']:5s}  start={format_timestamp(e['start_time'])}  "
              f"contact={format_timestamp(e['contact_time'])}  end={format_timestamp(e['end_time'])}  "
              f"min_dist={e['min_normalized_distance']:.3f}")


if __name__ == "__main__":
    main()
