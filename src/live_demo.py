"""Live hand-to-head touch detection on a webcam (or any video, played as a stream).

Runs the same pipeline as the batch detector -- MediaPipe pose + hand landmarks,
the head zone, the speed gate and the touch state machine -- but one frame at a
time, printing an alert the moment a touch is confirmed and again when it ends.
The frame-by-frame logic (`TouchDetector`) is the very code the batch path uses,
and tests/test_touch_logic.py checks that both give identical events.

Usage:
    python src/live_demo.py                          # default webcam
    python src/live_demo.py --source 1               # second camera
    python src/live_demo.py --record outputs/demo.mp4     # also save the annotated feed
    python src/live_demo.py --source data/video.mp4 --no-window --events-csv out.csv
    python src/live_demo.py --source data/video.mp4 --start 38 --duration 32 --record outputs/clip.mp4

Keys (window focused): q or Esc to quit.

Things to know when using it live
  * A touch alert appears after `--enter-frames` qualifying frames (default 3), so
    latency is about 3 frames -- 0.1 s at 30 fps, longer if the machine can only
    process a few frames per second.
  * The frame-count settings (enter 3 / exit 10) were chosen on a 30 fps recording.
    If live processing runs much slower than that, the same frame counts cover more
    real time, and behaviour will differ from the offline results.
  * The detector was evaluated offline on one recording; it has not been validated
    live. See the README for the limits of those results.
"""

import argparse
import os
import time
from collections import deque

import cv2
import mediapipe as mp

from behavior_events import DEFAULT_MIN_COUNT, DEFAULT_WINDOW_SECONDS, LiveRepetitionAggregator
from clip_writer import LiveClipWriter, write_behavior_events_csv
from head_touch_detector import (
    HAND_CONTACT_LANDMARK_IDS, TouchDetector, analyze_frame, apply_head_geometry,
    create_landmarkers, write_events_csv,
)
from timeutils import format_timestamp

GREEN = (80, 200, 80)
GREY = (90, 90, 90)
CYAN = (255, 220, 0)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
WHITE = (255, 255, 255)


def open_source(source: str):
    """Return (capture, is_live). A bare integer means a camera index."""
    if source.isdigit():
        # DirectShow opens much faster than the default backend on Windows.
        cap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(int(source))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # don't let stale frames queue up
        is_live = True
    else:
        cap = cv2.VideoCapture(source)
        is_live = False
    if not cap.isOpened():
        hint = ("Check the camera index and that no other app is using it, and that camera access "
                "is allowed for this terminal.") if is_live else "Check the file path."
        raise SystemExit(f"Could not open source {source!r}. {hint}")
    return cap, is_live


def draw_overlay(frame, head, rows, up, radius, touching_hands, touch_count, log, fps, now):
    h, w = frame.shape[:2]
    if head is not None:
        center, scale, _ = head
        c = (int(center[0]), int(center[1] - up * scale))
        cv2.circle(frame, c, int(radius * scale), CYAN, 2, cv2.LINE_AA)
    for row in rows:
        for i in HAND_CONTACT_LANDMARK_IDS:
            p = (int(row[f"c{i}_x"]), int(row[f"c{i}_y"]))
            cv2.circle(frame, p, 5, YELLOW if i == 0 else GREEN, -1, cv2.LINE_AA)
        text = f"{row['hand']} d={row['normalized_distance']:.2f}"
        if "velocity" in row and row["velocity"] != float("inf"):
            text += f" v={row['velocity']:.1f}"
        cv2.putText(frame, text, (int(row["c0_x"]) + 10, int(row["c0_y"]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)

    banner = f"TOUCH  ({', '.join(x.lower() for x in touching_hands)} hand)" if touching_hands else "no touch"
    cv2.rectangle(frame, (0, 0), (w, 36), GREEN if touching_hands else GREY, -1)
    cv2.putText(frame, banner, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0) if touching_hands else WHITE,
                2, cv2.LINE_AA)
    cv2.putText(frame, f"touches: {touch_count}   {fps:4.1f} fps", (w - 230, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0) if touching_hands else WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, f"t = {format_timestamp(now)}", (w - 130, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2, cv2.LINE_AA)
    for k, line in enumerate(log):
        cv2.putText(frame, line, (10, h - 12 - 20 * (len(log) - 1 - k)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description="Live hand-to-head touch detection.")
    ap.add_argument("--source", default="0", help="Camera index (default 0) or a video file path")
    ap.add_argument("--width", type=int, default=640,
                    help="Resize frames to this width before processing (0 = native). Smaller is faster.")
    ap.add_argument("--mirror", action="store_true", help="Flip horizontally (selfie view)")
    ap.add_argument("--record", default=None, help="Save the annotated feed to this .mp4 path")
    ap.add_argument("--record-fps", type=float, default=15.0,
                    help="Frame rate of the recording for a live camera (files use their own)")
    ap.add_argument("--events-csv", default=None, help="Write the detected events here on exit")
    ap.add_argument("--no-window", action="store_true", help="Headless: no preview window")
    ap.add_argument("--realtime", action="store_true",
                    help="For a file source, pace playback to the file's frame rate")
    ap.add_argument("--start", type=float, default=0,
                    help="File source only: start this many seconds into the video (times stay absolute)")
    ap.add_argument("--duration", type=float, default=0,
                    help="Stop after processing this many seconds (0 = until the end / until you quit)")
    ap.add_argument("--threshold", type=float, default=1.0)
    ap.add_argument("--enter-frames", type=int, default=3)
    ap.add_argument("--exit-frames", type=int, default=10)
    ap.add_argument("--contact-window", type=float, default=1.5)
    ap.add_argument("--velocity-threshold", type=float, default=1.5)
    ap.add_argument("--lost-timeout", type=float, default=1.0,
                    help="Close an open touch if its hand has not been seen for this many seconds "
                         "(a hand leaving the frame would otherwise leave the touch open forever)")
    ap.add_argument("--head-up", type=float, default=0.6)
    ap.add_argument("--head-radius", type=float, default=0.6)
    ap.add_argument("--bang-window", type=float, default=DEFAULT_WINDOW_SECONDS,
                    help="Max gap (seconds) between consecutive touches to still count as the same "
                         "bout, for head-banging detection")
    ap.add_argument("--bang-count", type=int, default=DEFAULT_MIN_COUNT,
                    help="Touches required within --bang-window to call a bout head-banging instead "
                         "of separate touches")
    ap.add_argument("--clip-dir", default=None,
                    help="Save one video clip per detected behavior event (touch or head-banging) "
                         "into this directory")
    ap.add_argument("--clip-preroll", type=float, default=1.0,
                    help="Seconds of buffered footage to include before a clip's event starts")
    ap.add_argument("--behavior-csv", default=None,
                    help="Write one row per detected behavior event here on exit, with a clip_path "
                         "column if --clip-dir is also given")
    args = ap.parse_args()

    cap, is_live = open_source(args.source)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_fps = args.record_fps if is_live else src_fps
    hand_lm, pose_lm = create_landmarkers()
    detector = TouchDetector(args.threshold, args.enter_frames, args.exit_frames,
                             args.contact_window, args.velocity_threshold,
                             lost_timeout_seconds=args.lost_timeout)
    aggregator = LiveRepetitionAggregator(window_seconds=args.bang_window, min_count=args.bang_count)
    clip_writer = LiveClipWriter(args.clip_dir, preroll_seconds=args.clip_preroll, fps=out_fps) \
        if args.clip_dir else None

    events, behaviors, log, touch_count, clip_index = [], [], deque(maxlen=5), 0, 0
    start_frame = int(args.start * src_fps) if (args.start and not is_live) else 0
    if start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    start_time = start_frame / src_fps
    writer, frame_idx, last_ts_ms, fps_ema = None, start_frame, -1, 0.0
    proc_ms_total = 0.0
    t0 = time.monotonic()
    print(f"Running on {'camera ' if is_live else ''}{args.source}. Press q or Esc in the window to quit."
          if not args.no_window else f"Running headless on {args.source}.")

    try:
        while True:
            loop_start = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break
            if args.mirror:
                frame = cv2.flip(frame, 1)
            if args.width and frame.shape[1] != args.width:
                frame = cv2.resize(frame, (args.width, int(frame.shape[0] * args.width / frame.shape[1])))
            h, w = frame.shape[:2]

            # Live: wall-clock time. File: frame index / fps, so results match the batch run.
            now = (time.monotonic() - t0) if is_live else frame_idx / src_fps  # absolute video time
            ts_ms = max(int(now * 1000), last_ts_ms + 1)  # MediaPipe needs strictly increasing timestamps
            last_ts_ms = ts_ms

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pose_res = pose_lm.detect_for_video(mp_image, ts_ms)
            hand_res = hand_lm.detect_for_video(mp_image, ts_ms)

            rows, head = analyze_frame(pose_res, hand_res, w, h, frame_idx, now)
            apply_head_geometry(rows, args.head_up, args.head_radius)
            for msg in detector.update(rows) + detector.tick(now):
                if msg["type"] == "start":
                    touch_count += 1
                    line = f"{format_timestamp(msg['detected_time'])} touch start ({msg['hand'].lower()})"
                    print(f"TOUCH START  {msg['hand'].lower():5s} hand  at {format_timestamp(msg['detected_time'])}"
                          f"  (began {format_timestamp(msg['start_time'])})")
                else:
                    e = msg["event"]
                    events.append(e)
                    aggregator.add(e)
                    line = f"{format_timestamp(e['end_time'])} touch end ({e['hand'].lower()})"
                    why = "  (hand left the frame)" if msg.get("reason") == "hand lost" else ""
                    print(f"TOUCH END    {e['hand'].lower():5s} hand  start={format_timestamp(e['start_time'])}"
                          f"  contact={format_timestamp(e['contact_time'])}  end={format_timestamp(e['end_time'])}{why}")
                log.append(line)

            finished_behaviors = aggregator.tick(now)

            proc_ms = (time.perf_counter() - loop_start) * 1000
            proc_ms_total += proc_ms
            inst_fps = 1000.0 / proc_ms if proc_ms > 0 else 0.0
            fps_ema = inst_fps if fps_ema == 0 else 0.9 * fps_ema + 0.1 * inst_fps

            draw_overlay(frame, head, rows, args.head_up, args.head_radius,
                         detector.touching_hands, touch_count, log, fps_ema, now)

            if clip_writer:
                clip_writer.step(frame, now, aggregator.active)
            for fb in finished_behaviors:
                clip_index += 1
                saved = clip_writer.save(fb, clip_index) if clip_writer else {**fb, "clip_path": ""}
                behaviors.append(saved)
                tap_note = f" ({saved['tap_count']} taps)" if saved["behavior"] != "head_touch" else ""
                clip_note = f"  clip={saved['clip_path']}" if saved.get("clip_path") else ""
                print(f"BEHAVIOR     {saved['behavior']}{tap_note}  start={format_timestamp(saved['start_time'])}"
                      f"  end={format_timestamp(saved['end_time'])}{clip_note}")

            if args.record:
                if writer is None:
                    os.makedirs(os.path.dirname(args.record) or ".", exist_ok=True)
                    writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"),
                                             args.record_fps if is_live else src_fps, (w, h))
                writer.write(frame)
            if not args.no_window:
                cv2.imshow("Head-touch detection (q to quit)", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            if args.realtime and not is_live:
                time.sleep(max(0.0, 1.0 / src_fps - (time.perf_counter() - loop_start)))

            frame_idx += 1
            if args.duration and (now - start_time) >= args.duration:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        hand_lm.close()
        pose_lm.close()
        cv2.destroyAllWindows()

    for e in detector.flush():  # a touch still open when the stream stopped
        events.append(e)
        aggregator.add(e)
        print(f"TOUCH END    {e['hand'].lower():5s} hand  (stream ended)  start={format_timestamp(e['start_time'])}"
              f"  contact={format_timestamp(e['contact_time'])}")
    events.sort(key=lambda e: e["start_time"])

    for fb in aggregator.flush():  # a behavior still open when the stream stopped
        clip_index += 1
        saved = clip_writer.save(fb, clip_index) if clip_writer else {**fb, "clip_path": ""}
        behaviors.append(saved)
        tap_note = f" ({saved['tap_count']} taps)" if saved["behavior"] != "head_touch" else ""
        clip_note = f"  clip={saved['clip_path']}" if saved.get("clip_path") else ""
        print(f"BEHAVIOR     {saved['behavior']}{tap_note}  (stream ended)  start={format_timestamp(saved['start_time'])}"
              f"  end={format_timestamp(saved['end_time'])}{clip_note}")

    frames_done = frame_idx - start_frame
    mean_ms = proc_ms_total / frames_done if frames_done else 0.0
    eff_fps = 1000.0 / mean_ms if mean_ms else 0.0
    print(f"\n{frames_done} frames, {mean_ms:.0f} ms/frame on average ({eff_fps:.1f} fps), "
          f"{len(events)} touch event(s).")
    if is_live and eff_fps and eff_fps < 20:
        print("Note: processing is slower than the 30 fps the frame-count settings were chosen at, so "
              "behaviour will differ from the offline results (see the docstring).")
    if args.events_csv:
        write_events_csv(args.events_csv, events)
        print(f"Events written to {args.events_csv}")
    if args.behavior_csv:
        write_behavior_events_csv(args.behavior_csv, behaviors)
        print(f"Behavior events written to {args.behavior_csv}")
    if args.record:
        print(f"Annotated feed saved to {args.record}")
    if args.clip_dir:
        print(f"{len(behaviors)} clip(s) saved to {args.clip_dir}")


if __name__ == "__main__":
    main()
