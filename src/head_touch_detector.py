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
SIGNAL_CSV_HEADER = ["frame", "time_sec", "hand", "normalized_distance", "head_scale_px", "head_source",
                     "wrist_x", "wrist_y"]

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


def compute_distance_signal(video_path: str, max_hands: int = 2):
    """Run pose + hand landmarkers over every frame and yield one dict per
    (frame, detected hand): {frame, time_sec, hand, normalized_distance,
    head_scale_px, head_source}. Frames with no usable head estimate or no
    detected hand are skipped (no row emitted)."""
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

            if pose_result.pose_landmarks and hand_result.hand_landmarks:
                head = head_center_and_scale(pose_result.pose_landmarks[0], width, height)
                if head is not None:
                    center, scale, source = head
                    for hand_landmarks, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
                        hand_label = handedness[0].category_name  # 'Left' / 'Right'
                        min_px_dist = min(
                            _dist(_to_px(hand_landmarks[i], width, height), center)
                            for i in HAND_CONTACT_LANDMARK_IDS
                        )
                        wrist_x, wrist_y = _to_px(hand_landmarks[WRIST_ID], width, height)
                        yield {
                            "frame": frame_idx,
                            "time_sec": frame_idx / fps,
                            "hand": hand_label,
                            "normalized_distance": min_px_dist / scale,
                            "head_scale_px": scale,
                            "head_source": source,
                            "wrist_x": wrist_x,
                            "wrist_y": wrist_y,
                        }
            frame_idx += 1
    finally:
        cap.release()
        hand_landmarker.close()
        pose_landmarker.close()


def add_velocities(rows) -> None:
    """Annotate time-sorted rows of ONE hand with a `velocity` key, in
    head-widths per second (wrist speed / head_scale_px).

    The wrist is used as the anchor rather than the closest-of-several
    contact landmarks used for distance, because that "closest" landmark
    can flip between fingertips frame to frame and fake motion on a hand
    that is actually still. Step speeds are smoothed over a short trailing
    window since single-frame landmark jitter is comparable to real slow
    motion. A row with no usable predecessor (first row, or a gap longer
    than MAX_VELOCITY_GAP_SEC) gets `inf`, i.e. never counts as slow.
    """
    step_speeds = []
    for i, row in enumerate(rows):
        speed = float("inf")
        if i > 0:
            prev = rows[i - 1]
            dt = row["time_sec"] - prev["time_sec"]
            if 0 < dt <= MAX_VELOCITY_GAP_SEC:
                px = _dist((row["wrist_x"], row["wrist_y"]), (prev["wrist_x"], prev["wrist_y"]))
                speed = px / dt / row["head_scale_px"]
        step_speeds.append(speed)
        window = step_speeds[-VELOCITY_SMOOTHING_ROWS:]
        row["velocity"] = sum(window) / len(window)


def extract_events(signal_rows, threshold: float, enter_frames: int, exit_frames: int,
                    contact_window_seconds: float = 1.5, velocity_threshold: float = None):
    """State machine per hand label -> list of event dicts with keys
    start_time, contact_time, end_time, hand (all in seconds), plus
    min_normalized_distance for diagnostics.

    `contact_window_seconds` bounds how far past `start_time` we look for
    the "contact" frame (the local minimum of the distance signal). This
    matters because the hand can linger near the threshold for a long
    stretch (fidgeting, resting near the face after a real touch) without
    the exit debounce firing, which keeps a single event "open" for much
    longer than the actual touch lasted. Searching the *whole* open
    stretch for a global minimum can then pick a later, unrelated dip as
    "contact". Ground-truth touches in this project are consistently
    under ~1.5s from start to end, so that's the default search window;
    `end_time` (and the exit debounce that produces it) is unaffected and
    can still extend well past it.

    `velocity_threshold` (head-widths/sec, None = off) adds a second gate:
    a frame only counts as "touching" if the hand is close AND slow. A real
    touch decelerates to near-standstill at the head, while a gesture or
    hair-flick passes through the close zone at speed. Because the gate
    feeds the same enter/exit debounce, `enter_frames` doubles as the
    dwell-time requirement.
    """
    by_hand = {}
    for row in signal_rows:
        by_hand.setdefault(row["hand"], []).append(row)

    events = []
    for hand_label, rows in by_hand.items():
        rows.sort(key=lambda r: r["frame"])
        if velocity_threshold is not None:
            add_velocities(rows)
        state = "idle"
        consecutive_below = 0
        consecutive_above = 0
        start_row = None
        best_row = None  # min normalized_distance row within the contact search window

        for idx, row in enumerate(rows):
            below = row["normalized_distance"] <= threshold
            if velocity_threshold is not None:
                below = below and row["velocity"] <= velocity_threshold

            if state == "idle":
                consecutive_below = consecutive_below + 1 if below else 0
                if consecutive_below >= enter_frames:
                    # Back-date the start to where the run of "below" frames began.
                    start_idx = idx - enter_frames + 1
                    start_row = rows[max(start_idx, 0)]
                    best_row = row
                    state = "touching"
                    consecutive_above = 0
            else:  # state == "touching"
                if (row["time_sec"] - start_row["time_sec"] <= contact_window_seconds
                        and row["normalized_distance"] < best_row["normalized_distance"]):
                    best_row = row
                consecutive_above = consecutive_above + 1 if not below else 0
                if consecutive_above >= exit_frames:
                    end_idx = idx - exit_frames
                    end_row = rows[max(end_idx, 0)]
                    events.append({
                        "hand": hand_label,
                        "start_time": start_row["time_sec"],
                        "contact_time": best_row["time_sec"],
                        # Back-dating the end by exit_frames can land before the
                        # contact frame on short events; an event can't end before it touches.
                        "end_time": max(end_row["time_sec"], best_row["time_sec"]),
                        "min_normalized_distance": best_row["normalized_distance"],
                    })
                    state = "idle"
                    consecutive_below = 0
                    start_row = None
                    best_row = None

        if state == "touching":
            # Video ended while still in contact; close the event at the last row.
            events.append({
                "hand": hand_label,
                "start_time": start_row["time_sec"],
                "contact_time": best_row["time_sec"],
                "end_time": rows[-1]["time_sec"],
                "min_normalized_distance": best_row["normalized_distance"],
            })

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
            })
    return rows


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
    parser.add_argument("--velocity-threshold", type=float, default=0.5,
                         help="Max wrist speed (head-widths/sec) for a frame to count as touching; "
                              "omit to disable the velocity gate")
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
