"""Save an annotated frame at each detected event's contact time.

Draws what the detector "sees" -- the head circle (radius = head_scale, so
the edge of the circle is normalized distance 1.0), the head keypoints, the
wrist and fingertips of each detected hand, and the closest hand landmark
with a line to the head center -- and labels the frame as a true positive
(matched to a ground-truth event) or a false positive. This is for human
review: it is how false positives can be inspected without re-watching the
whole video.

Landmarks are re-computed on a short warm-up window before each target frame
(VIDEO-mode tracking needs a few frames of context), so drawn values can
differ slightly from the full-run signal; the labelled distance/speed are
taken from the cached signal CSV when one is given.

Usage:
    python src/visualize_events.py data/test_video_task.mp4 outputs/detected_events.csv \\
        annotations/test_video_task_annotations.csv --signal outputs/distance_signal.csv \\
        --out results/frames
"""

import argparse
import os

import cv2
import mediapipe as mp

from evaluate import load_events, match_events
from head_touch_detector import (
    HAND_CONTACT_LANDMARK_IDS, HEAD_LANDMARK_IDS, _dist, _to_px, add_velocities,
    create_landmarkers, head_center_and_scale, load_signal_csv,
)
from timeutils import format_timestamp

GREEN = (80, 200, 80)
ORANGE = (0, 140, 255)
CYAN = (255, 220, 0)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
FINGERTIP_IDS = HAND_CONTACT_LANDMARK_IDS[1:]  # everything except the wrist (id 0)


def render_frame(video_path, target_frame, fps, warmup_frames, width, height):
    """Run the landmarkers over a warm-up window ending at target_frame and
    return (frame_bgr, head, hands) where head = (center, scale, source) or
    None, and hands = [(label, landmarks)] for the target frame."""
    cap = cv2.VideoCapture(video_path)
    start = max(0, target_frame - warmup_frames)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    hand_lm, pose_lm = create_landmarkers()
    frame_bgr, head, hands = None, None, []
    try:
        for idx in range(start, target_frame + 1):
            ok, frame = cap.read()
            if not ok:
                break
            ts = int(round(idx / fps * 1000))
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pose_res = pose_lm.detect_for_video(mp_image, ts)
            hand_res = hand_lm.detect_for_video(mp_image, ts)
            if idx == target_frame:
                frame_bgr = frame
                if pose_res.pose_landmarks:
                    head = head_center_and_scale(pose_res.pose_landmarks[0], width, height)
                    head_points = [_to_px(pose_res.pose_landmarks[0][i], width, height)
                                   for i in HEAD_LANDMARK_IDS
                                   if pose_res.pose_landmarks[0][i].visibility >= 0.5]
                else:
                    head_points = []
                hands = [(h[0].category_name, lm)
                         for lm, h in zip(hand_res.hand_landmarks, hand_res.handedness)]
    finally:
        cap.release()
        hand_lm.close()
        pose_lm.close()
    return frame_bgr, head, head_points, hands


def draw(frame, head, head_points, hands, width, height, banner, banner_color, signal_lookup, frame_idx):
    if head is not None:
        center, scale, source = head
        c = (int(center[0]), int(center[1]))
        cv2.circle(frame, c, int(scale), CYAN, 2, cv2.LINE_AA)
        cv2.circle(frame, c, 4, CYAN, -1, cv2.LINE_AA)
        for p in head_points:
            cv2.circle(frame, (int(p[0]), int(p[1])), 4, WHITE, -1, cv2.LINE_AA)
    for label, lm in hands:
        pts = {i: _to_px(lm[i], width, height) for i in HAND_CONTACT_LANDMARK_IDS}
        for i, p in pts.items():
            color = YELLOW if i == 0 else GREEN
            cv2.circle(frame, (int(p[0]), int(p[1])), 6, color, -1, cv2.LINE_AA)
        if head is not None:
            closest = min(pts.values(), key=lambda p: _dist(p, head[0]))
            cv2.line(frame, (int(closest[0]), int(closest[1])),
                     (int(head[0][0]), int(head[0][1])), RED, 2, cv2.LINE_AA)
            cv2.circle(frame, (int(closest[0]), int(closest[1])), 10, RED, 2, cv2.LINE_AA)
        row = signal_lookup.get((frame_idx, label))
        info = f"{label}"
        if row is not None:
            info += f"  dist={row['normalized_distance']:.2f}"
            if "velocity" in row and row["velocity"] != float("inf"):
                info += f"  speed={row['velocity']:.2f}"
        wx, wy = pts[0]
        cv2.putText(frame, info, (int(wx) + 12, int(wy) - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, WHITE, 2, cv2.LINE_AA)
    cv2.rectangle(frame, (0, 0), (width, 44), banner_color, -1)
    cv2.putText(frame, banner, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 2, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="Save annotated frames at detected contact times.")
    parser.add_argument("video")
    parser.add_argument("detected_csv")
    parser.add_argument("ground_truth_csv")
    parser.add_argument("--signal", default=None, help="Cached signal CSV, for labelled dist/speed")
    parser.add_argument("--out", default="results/frames")
    parser.add_argument("--tolerance", type=float, default=0.5)
    parser.add_argument("--warmup-seconds", type=float, default=1.0)
    args = parser.parse_args()

    detected = load_events(args.detected_csv)
    ground_truth = load_events(args.ground_truth_csv)
    matches, _, _ = match_events(ground_truth, detected, args.tolerance)
    match_by_det = {id(det): (gt, diff) for gt, det, diff in matches}

    signal_lookup = {}
    if args.signal:
        rows = load_signal_csv(args.signal)
        by_hand = {}
        for r in rows:
            by_hand.setdefault(r["hand"], []).append(r)
        for hand_rows in by_hand.values():
            hand_rows.sort(key=lambda r: r["frame"])
            add_velocities(hand_rows)
        signal_lookup = {(r["frame"], r["hand"]): r for r in rows}

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    os.makedirs(args.out, exist_ok=True)
    for det in detected:
        target_frame = int(round(det["contact_time"] * fps))
        frame, head, head_points, hands = render_frame(
            args.video, target_frame, fps, int(args.warmup_seconds * fps), width, height)
        if frame is None:
            print(f"det#{det['event_id']}: could not read frame {target_frame}")
            continue
        stamp = format_timestamp(det["contact_time"])
        if id(det) in match_by_det:
            gt, diff = match_by_det[id(det)]
            kind = "TP"
            banner = (f"TRUE POSITIVE  det#{det['event_id']} @ {stamp}  "
                      f"(matches GT#{gt['event_id']}, {diff:.2f}s off)")
            color = GREEN
        else:
            kind = "FP"
            banner = f"FALSE POSITIVE  det#{det['event_id']} @ {stamp}  (no matching ground truth)"
            color = ORANGE
        draw(frame, head, head_points, hands, width, height, banner, color, signal_lookup, target_frame)
        name = f"{kind}_det{int(det['event_id']):02d}_{stamp.replace(':', 'm')}s.jpg"
        cv2.imwrite(os.path.join(args.out, name), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"saved {name}")


if __name__ == "__main__":
    main()
