"""Cut and save a video clip for each detected behavior event.

Two paths, since a finished recording and a live stream need different
strategies:
  * `export_clips` -- batch: the whole file is already on disk, so each clip
    is cut by seeking directly to (start_time - preroll) and reading through
    (end_time + postroll).
  * `LiveClipWriter` -- live: frames only exist as they arrive, so a small
    rolling buffer holds the last `preroll_seconds` at all times, and frames
    are appended to an in-memory "active" clip for as long as
    LiveRepetitionAggregator reports a cluster open (see its docstring for
    why that already covers the postroll -- no separate postroll buffer is
    needed live).

Both write plain mp4v .mp4 files (the same codec `live_demo.py --record`
already uses) plus one metadata CSV row per clip.
"""

import csv
import os

import cv2

from timeutils import format_timestamp

BEHAVIOR_EVENTS_CSV_HEADER = ["event_id", "behavior", "start_time", "contact_time", "end_time",
                              "hand", "tap_count", "min_normalized_distance", "clip_path"]


def write_behavior_events_csv(path, events) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(BEHAVIOR_EVENTS_CSV_HEADER)
        for i, e in enumerate(events, start=1):
            writer.writerow([
                i, e["behavior"], format_timestamp(e["start_time"]), format_timestamp(e["contact_time"]),
                format_timestamp(e["end_time"]), e["hand"].lower(), e.get("tap_count", 1),
                f"{e['min_normalized_distance']:.3f}", e.get("clip_path", ""),
            ])


def _clip_name(index, event):
    return f"{index:03d}_{event['behavior']}_{format_timestamp(event['start_time']).replace(':', 'm')}s.mp4"


def export_clips(video_path, behavior_events, out_dir, preroll=1.0, postroll=1.0):
    """Batch: cut one clip per behavior event out of a finished recording.

    Returns the same events, each with a "clip_path" key added (empty string
    if no frames could be read for that event).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    os.makedirs(out_dir, exist_ok=True)

    out_events = []
    try:
        for i, event in enumerate(behavior_events, start=1):
            start_f = max(int((event["start_time"] - preroll) * fps), 0)
            end_f = min(int((event["end_time"] + postroll) * fps), max(frame_count - 1, 0))
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
            clip_path = os.path.join(out_dir, _clip_name(i, event))
            writer = None
            for _ in range(start_f, end_f + 1):
                ok, frame = cap.read()
                if not ok:
                    break
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(clip_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                writer.write(frame)
            if writer is not None:
                writer.release()
            out_events.append({**event, "clip_path": clip_path if writer is not None else ""})
    finally:
        cap.release()
    return out_events


class LiveClipWriter:
    """Rolling pre-roll buffer + one saved clip per finalized behavior event.

    Call step() every frame with the current frame, its timestamp, and
    whether a LiveRepetitionAggregator cluster is currently open. Call
    save() with the finished event as soon as the aggregator reports it --
    its frames are already buffered by then.
    """

    def __init__(self, out_dir, preroll_seconds=1.0, fps=30.0):
        self.out_dir = out_dir
        self.preroll_seconds = preroll_seconds
        self.fps = fps
        self._ring = []    # [(frame, time_sec), ...], trimmed to the last preroll_seconds
        self._active = []  # frames for the clip currently being recorded

    def step(self, frame, now, cluster_active: bool) -> None:
        self._ring.append((frame.copy(), now))
        while self._ring and now - self._ring[0][1] > self.preroll_seconds:
            self._ring.pop(0)
        if cluster_active:
            if not self._active:
                self._active = list(self._ring)  # seed with what's already buffered (the pre-roll)
            else:
                self._active.append((frame.copy(), now))

    def save(self, event, index: int):
        """Write the buffered active clip for `event`, reset it, and return
        the event with a "clip_path" key added (empty string if there was
        nothing buffered)."""
        frames, self._active = self._active, []
        if not frames:
            return {**event, "clip_path": ""}
        os.makedirs(self.out_dir, exist_ok=True)
        clip_path = os.path.join(self.out_dir, _clip_name(index, event))
        h, w = frames[0][0].shape[:2]
        writer = cv2.VideoWriter(clip_path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
        for f, _ in frames:
            writer.write(f)
        writer.release()
        return {**event, "clip_path": clip_path}
