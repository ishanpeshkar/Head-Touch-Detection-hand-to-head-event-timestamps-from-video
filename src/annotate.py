"""Phase 2: interactive ground-truth annotation tool.

Lets you scrub through a video frame-by-frame and log head-touch events
(start / contact / end) to a CSV file for later evaluation of the detector.

Usage:
    python src/annotate.py data/test_video.mp4 annotations/test_video_annotations.csv
    python src/annotate.py data/test_video.mp4 annotations/test_video_annotations.csv --start 02:30.00

Controls (video window must be focused):
    space       play / pause
    a / d       step one frame back / forward (while paused)
    j / l       jump ~1 second back / forward (while paused)
    s           mark START of a head-touch event at current timestamp
    c           mark CONTACT (moment of touch) at current timestamp
    e           mark END of the event at current timestamp, then prompts
                in the console for hand (left/right/both) and a note,
                and appends one row to the CSV
    r           reset pending start/contact/end marks without saving
    q / ESC     quit (already-saved rows remain in the CSV)
"""

import argparse
import csv
import os

import cv2

from timeutils import format_timestamp, parse_timestamp

CSV_HEADER = ["event_id", "start_time", "contact_time", "end_time", "hand", "notes"]


def load_existing_event_count(csv_path: str) -> int:
    if not os.path.exists(csv_path):
        return 0
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    return max(len(rows) - 1, 0)


def append_event(csv_path: str, event_id: int, start, contact, end, hand: str, notes: str) -> None:
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(CSV_HEADER)
        writer.writerow([event_id, start, contact, end, hand, notes])


def main():
    parser = argparse.ArgumentParser(description="Interactive head-touch ground-truth annotator.")
    parser.add_argument("video", help="Path to the input video file")
    parser.add_argument("csv", help="Path to the annotations CSV to append to")
    parser.add_argument("--start", default="00:00.00",
                        help="Timestamp (MM:SS.ss) to open the video at, e.g. 02:30.00")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    jump_frames = max(int(round(fps)), 1)

    os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
    next_event_id = load_existing_event_count(args.csv) + 1

    print(__doc__)
    print(f"Loaded {args.video}: {frame_count} frames @ {fps:.2f} fps")
    print(f"Appending to {args.csv} (next event_id = {next_event_id})\n")

    current_frame = int(parse_timestamp(args.start) * fps)
    playing = False
    pending_start = None
    pending_contact = None

    window = "Annotator (space=play/pause, s/c/e=mark, q=quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    while True:
        current_frame = max(0, min(current_frame, frame_count - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ok, frame = cap.read()
        if not ok:
            break

        timestamp = current_frame / fps
        overlay = frame.copy()
        status = "PLAYING" if playing else "PAUSED"
        text_lines = [
            f"frame {current_frame}/{frame_count - 1}  t={format_timestamp(timestamp)}  [{status}]",
            f"pending: start={pending_start}  contact={pending_contact}",
        ]
        for i, line in enumerate(text_lines):
            cv2.putText(overlay, line, (10, 25 + i * 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow(window, overlay)

        wait_ms = int(1000 / fps) if playing else 30
        key = cv2.waitKey(wait_ms) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            playing = not playing
        elif key == ord("a") and not playing:
            current_frame -= 1
        elif key == ord("d") and not playing:
            current_frame += 1
        elif key == ord("j") and not playing:
            current_frame -= jump_frames
        elif key == ord("l") and not playing:
            current_frame += jump_frames
        elif key == ord("s"):
            pending_start = format_timestamp(timestamp)
            print(f"start marked: {pending_start}")
        elif key == ord("c"):
            pending_contact = format_timestamp(timestamp)
            print(f"contact marked: {pending_contact}")
        elif key == ord("r"):
            pending_start = None
            pending_contact = None
            print("pending marks reset")
        elif key == ord("e"):
            end_time = format_timestamp(timestamp)
            if pending_start is None or pending_contact is None:
                print("cannot log event: need both 's' (start) and 'c' (contact) marked first")
                continue
            hand = input("  hand (left/right/both) [right]: ").strip() or "right"
            notes = input("  notes: ").strip()
            append_event(args.csv, next_event_id, pending_start, pending_contact, end_time, hand, notes)
            print(f"event {next_event_id} saved: {pending_start} -> {pending_contact} -> {end_time}\n")
            next_event_id += 1
            pending_start = None
            pending_contact = None

        if playing:
            current_frame += 1
            if current_frame >= frame_count:
                playing = False
                current_frame = frame_count - 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
