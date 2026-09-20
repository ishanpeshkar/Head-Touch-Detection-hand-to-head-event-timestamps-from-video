"""Tests for the touch event logic (no video or camera needed).

Run from the project root:
    python -m unittest discover -s tests -v

What is checked
  * The frame-by-frame TouchDetector (used by live mode) gives exactly the same
    events as the original whole-recording implementation (kept below as
    `legacy_extract_events`) across a grid of settings on the real signal in
    results/distance_signal.csv.
  * The committed results/detected_events.csv is what the current defaults produce.
  * Synthetic scenarios for the behaviours that matter on a live feed.
"""

import csv
import itertools
import os
import sys
import unittest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from head_touch_detector import (  # noqa: E402
    TouchDetector, VelocityTracker, apply_head_geometry, extract_events, load_signal_csv,
)
from timeutils import format_timestamp  # noqa: E402

SIGNAL = os.path.join(ROOT, "results", "distance_signal.csv")
COMMITTED_EVENTS = os.path.join(ROOT, "results", "detected_events.csv")

MAX_VELOCITY_GAP_SEC = 0.25
VELOCITY_SMOOTHING_ROWS = 3


# --------------------------------------------------------------------------
# Reference: the original whole-recording implementation, before the logic was
# made incremental. Kept verbatim so the new code can be compared against it.
# --------------------------------------------------------------------------
def _dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def legacy_add_velocities(rows):
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


def legacy_extract_events(signal_rows, threshold, enter_frames, exit_frames,
                          contact_window_seconds=1.5, velocity_threshold=None):
    by_hand = {}
    for row in signal_rows:
        by_hand.setdefault(row["hand"], []).append(row)

    events = []
    for hand_label, rows in by_hand.items():
        rows.sort(key=lambda r: r["frame"])
        if velocity_threshold is not None:
            legacy_add_velocities(rows)
        state = "idle"
        consecutive_below = 0
        consecutive_above = 0
        start_row = None
        best_row = None

        for idx, row in enumerate(rows):
            below = row["normalized_distance"] <= threshold
            if velocity_threshold is not None:
                below = below and row["velocity"] <= velocity_threshold

            if state == "idle":
                consecutive_below = consecutive_below + 1 if below else 0
                if consecutive_below >= enter_frames:
                    start_idx = idx - enter_frames + 1
                    start_row = rows[max(start_idx, 0)]
                    best_row = row
                    state = "touching"
                    consecutive_above = 0
            else:
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
                        "end_time": max(end_row["time_sec"], best_row["time_sec"]),
                        "min_normalized_distance": best_row["normalized_distance"],
                    })
                    state = "idle"
                    consecutive_below = 0
                    start_row = None
                    best_row = None

        if state == "touching":
            events.append({
                "hand": hand_label,
                "start_time": start_row["time_sec"],
                "contact_time": best_row["time_sec"],
                "end_time": rows[-1]["time_sec"],
                "min_normalized_distance": best_row["normalized_distance"],
            })

    events.sort(key=lambda e: e["start_time"])
    return events


# --------------------------------------------------------------------------
def fresh_rows(up=0.6, radius=0.6):
    rows = load_signal_csv(SIGNAL)
    apply_head_geometry(rows, up, radius)
    return rows


def synthetic_rows(spec, hand="Right", fps=30.0, head_scale=100.0):
    """spec: list of (distance, wrist_x) per frame, or None for a frame with no
    detection. Wrist moves along x; speed = dx * fps / head_scale head-widths/sec."""
    rows = []
    for frame, item in enumerate(spec):
        if item is None:
            continue
        distance, wrist_x = item
        rows.append({"frame": frame, "time_sec": frame / fps, "hand": hand,
                     "normalized_distance": distance, "head_scale_px": head_scale,
                     "wrist_x": wrist_x, "wrist_y": 0.0})
    return rows


def run_stream(rows, **kw):
    """Feed rows one frame at a time, as the live loop does; return (messages, events)."""
    detector = TouchDetector(**kw)
    messages = []
    for frame in sorted({r["frame"] for r in rows}):
        messages.extend(detector.update([r for r in rows if r["frame"] == frame]))
    events = [m["event"] for m in messages if m["type"] == "end"] + detector.flush()
    return messages, events


DEFAULTS = dict(threshold=1.0, enter_frames=3, exit_frames=10,
                contact_window_seconds=1.5, velocity_threshold=1.5)


class EquivalenceWithOriginal(unittest.TestCase):
    def test_grid_of_settings_matches_original_implementation(self):
        grid = itertools.product([0.9, 1.0], [2, 3], [5, 10], [None, 0.5, 1.5], [(0.0, 1.0), (0.6, 0.6)])
        checked = 0
        for thr, enter, exit_, vel, (up, rad) in grid:
            new = extract_events(fresh_rows(up, rad), thr, enter, exit_, velocity_threshold=vel)
            old = legacy_extract_events(fresh_rows(up, rad), thr, enter, exit_, velocity_threshold=vel)
            self.assertEqual(new, old, f"differs at thr={thr} enter={enter} exit={exit_} vel={vel} head=({up},{rad})")
            checked += 1
        self.assertEqual(checked, 48)

    def test_velocity_tracker_matches_original(self):
        rows = [r for r in load_signal_csv(SIGNAL) if r["hand"] == "Right"]
        rows.sort(key=lambda r: r["frame"])
        expected = [dict(r) for r in rows]
        legacy_add_velocities(expected)
        tracker = VelocityTracker()
        for row, exp in zip(rows, expected):
            self.assertEqual(tracker.update(row), exp["velocity"])

    def test_committed_results_are_what_current_defaults_produce(self):
        events = extract_events(fresh_rows(), **DEFAULTS)
        with open(COMMITTED_EVENTS, newline="") as f:
            committed = list(csv.DictReader(f))
        self.assertEqual(len(events), len(committed))
        for e, c in zip(events, committed):
            self.assertEqual(format_timestamp(e["start_time"]), c["start_time"])
            self.assertEqual(format_timestamp(e["contact_time"]), c["contact_time"])
            self.assertEqual(format_timestamp(e["end_time"]), c["end_time"])

    def test_frame_by_frame_stream_equals_batch(self):
        rows = fresh_rows()
        _, streamed = run_stream(rows, **DEFAULTS)
        streamed.sort(key=lambda e: e["start_time"])
        self.assertEqual(streamed, extract_events(fresh_rows(), **DEFAULTS))

    def test_start_alert_comes_before_its_end_and_is_confirmed_after_enter_frames(self):
        messages, _ = run_stream(fresh_rows(), **DEFAULTS)
        starts = [m for m in messages if m["type"] == "start"]
        ends = [m["event"] for m in messages if m["type"] == "end"]
        self.assertGreaterEqual(len(starts), len(ends))
        for s in starts:
            self.assertGreaterEqual(s["detected_time"], s["start_time"])
            # confirmed after at most a couple of dropped frames' worth of time
            self.assertLess(s["detected_time"] - s["start_time"], 0.5)


class SyntheticScenarios(unittest.TestCase):
    """Head-widths/sec = dx_px_per_frame * 30 / 100, so 1 px/frame = 0.3 head-widths/sec."""

    def test_hand_held_still_near_head_is_one_touch(self):
        rows = synthetic_rows([(0.5, 0.0)] * 30 + [(2.0, 0.0)] * 20)
        _, events = run_stream(rows, **DEFAULTS)
        self.assertEqual(len(events), 1)

    def test_fast_sweep_through_the_zone_is_not_a_touch(self):
        # 20 px/frame = 6 head-widths/sec, well above the 1.5 limit
        rows = synthetic_rows([(0.5, 20.0 * i) for i in range(30)] + [(2.0, 600.0)] * 20)
        _, events = run_stream(rows, **DEFAULTS)
        self.assertEqual(events, [])

    def test_two_frame_dip_is_ignored_by_debounce(self):
        rows = synthetic_rows([(2.0, 0.0)] * 10 + [(0.5, 0.0)] * 2 + [(2.0, 0.0)] * 20)
        _, events = run_stream(rows, **DEFAULTS)
        self.assertEqual(events, [])

    def test_hand_lost_for_a_few_frames_mid_touch_keeps_one_touch(self):
        spec = [(0.5, 0.0)] * 10 + [None] * 6 + [(0.5, 0.0)] * 10 + [(2.0, 0.0)] * 20
        _, events = run_stream(rows := synthetic_rows(spec), **DEFAULTS)
        self.assertEqual(len(events), 1)

    def test_touch_still_open_when_stream_ends_is_flushed(self):
        rows = synthetic_rows([(2.0, 0.0)] * 5 + [(0.5, 0.0)] * 20)
        messages, events = run_stream(rows, **DEFAULTS)
        self.assertEqual([m["type"] for m in messages], ["start"])  # never ended on its own
        self.assertEqual(len(events), 1)                            # closed by flush()

    def test_start_is_back_dated_to_first_qualifying_frame(self):
        rows = synthetic_rows([(2.0, 0.0)] * 5 + [(0.5, 0.0)] * 20 + [(2.0, 0.0)] * 20)
        messages, events = run_stream(rows, **DEFAULTS)
        start = next(m for m in messages if m["type"] == "start")
        # Frame 5 is the first frame inside the zone. The unknown speed of the very
        # first row has left the 3-frame smoothing window by then, so it qualifies:
        # start is back-dated to frame 5 and confirmed 2 frames later (3rd qualifying frame).
        self.assertAlmostEqual(start["start_time"], 5 / 30.0)
        self.assertAlmostEqual(start["detected_time"], 7 / 30.0)
        self.assertAlmostEqual(events[0]["start_time"], 5 / 30.0)

    def test_live_timeout_closes_a_touch_when_the_hand_leaves_the_frame(self):
        rows = synthetic_rows([(2.0, 0.0)] * 5 + [(0.5, 0.0)] * 20)  # hand last seen at frame 24
        detector = TouchDetector(**DEFAULTS, lost_timeout_seconds=1.0)
        for r in rows:
            detector.update([r])
        last_seen = rows[-1]["time_sec"]
        self.assertEqual(detector.tick(last_seen + 0.9), [])  # not yet
        self.assertEqual(detector.touching_hands, ["Right"])
        messages = detector.tick(last_seen + 1.1)
        self.assertEqual([m["type"] for m in messages], ["end"])
        self.assertEqual(messages[0]["reason"], "hand lost")
        self.assertAlmostEqual(messages[0]["event"]["end_time"], last_seen)
        self.assertEqual(detector.touching_hands, [])
        self.assertEqual(detector.flush(), [])  # nothing left open

    def test_timeout_is_off_by_default_so_batch_behaviour_is_unchanged(self):
        rows = synthetic_rows([(2.0, 0.0)] * 5 + [(0.5, 0.0)] * 20)
        detector = TouchDetector(**DEFAULTS)
        for r in rows:
            detector.update([r])
        self.assertEqual(detector.tick(rows[-1]["time_sec"] + 60.0), [])
        self.assertEqual(detector.touching_hands, ["Right"])

    def test_two_hands_are_tracked_independently(self):
        right = synthetic_rows([(0.5, 0.0)] * 30 + [(2.0, 0.0)] * 20, hand="Right")
        left = synthetic_rows([(2.0, 0.0)] * 50, hand="Left")
        _, events = run_stream(right + left, **DEFAULTS)
        self.assertEqual([e["hand"] for e in events], ["Right"])


if __name__ == "__main__":
    unittest.main()
