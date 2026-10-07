"""Stage 5: run the whole pipeline on a video with the trained model and cut a clip per event.

Usage:
    python behavior_lab/src/detect.py data/new/test_video.mp4
    python behavior_lab/src/detect.py some_video.mp4 --enroll 5 20     # face-enrollment window (seconds)
"""

import argparse
import csv
import os
import pickle
import subprocess

import cv2
import imageio_ffmpeg

import classifier
import extract
import face_focus
import features

OUT_DIR = os.path.join(extract.LAB_DIR, "outputs", "runs")


def load_model():
    path = os.path.join(classifier.MODEL_DIR, "behavior_model.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError("No trained model yet - run classifier.py on an annotated video first.")
    with open(path, "rb") as f:
        return pickle.load(f)


def run(video_path, enroll_window=None, progress=None, min_conf=0.5):
    """Returns dict: events, enroll_window, data, subject, subject_frames_pct."""
    bundle = load_model()
    data = extract.load_or_extract(video_path, progress)
    if enroll_window is None:
        enroll_window = face_focus.auto_enrollment_window(data)
    identity = face_focus.enroll(data, *enroll_window)
    subject = face_focus.find_subject(data, identity)
    sig = features.frame_signals(data, subject)
    X, starts, _ = features.window_features(sig, data["fps"], win_sec=bundle["win_sec"])
    proba = bundle["model"].predict_proba(X)
    full = classifier.np.zeros((len(X), len(classifier.CLASSES)))
    full[:, bundle["model"].classes_] = proba
    events = classifier.windows_to_events(starts, bundle["win_sec"], full, min_conf=min_conf)
    pct = 100.0 * sum(s[0] is not None for s in subject) / max(1, len(subject))
    return {"events": events, "enroll_window": enroll_window, "data": data, "subject": subject,
            "subject_frames_pct": pct}


def enrollment_thumbnail(video_path, data, window):
    """Face crop from the middle of the enrollment window, so the user can see who was learned."""
    fps = data["fps"]
    mid = int(sum(window) / 2 * fps)
    cap = cv2.VideoCapture(video_path)
    for i in range(mid, min(mid + 60, len(data["frames"]))):
        fr = data["frames"][i]
        if len(fr["faces"]) == 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, img = cap.read()
            cap.release()
            if ok:
                x, y, w, h = fr["faces"][0][:4].astype(int)
                pad = int(0.3 * w)
                return cv2.cvtColor(img[max(0, y - pad): y + h + pad, max(0, x - pad): x + w + pad], cv2.COLOR_BGR2RGB)
    cap.release()
    return None


def export_clips(video_path, events, out_dir, pad=1.0):
    """Cut one browser-playable H.264 clip per event. Returns the events with a 'clip' path added."""
    os.makedirs(out_dir, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    out = []
    for k, e in enumerate(events, 1):
        start = max(0.0, e["start"] - pad)
        length = (e["end"] + pad) - start
        name = f"event{k:02d}_{e['behavior']}_{int(e['start'] // 60):02d}m{e['start'] % 60:05.2f}s.mp4"
        path = os.path.join(out_dir, name)
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-ss", f"{start:.2f}", "-i", video_path, "-t", f"{length:.2f}",
                        "-c:v", "libx264", "-preset", "fast", "-crf", "24", "-pix_fmt", "yuv420p", "-an", path], check=True)
        out.append({**e, "clip": path})
    return out


def write_events_csv(events, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["event_id", "start_time", "end_time", "duration_sec", "behavior", "confidence", "clip"])
        for k, e in enumerate(events, 1):
            w.writerow([k, classifier.fmt_time(e["start"]), classifier.fmt_time(e["end"]),
                        f"{e['end'] - e['start']:.2f}", e["behavior"], f"{e['confidence']:.2f}", e.get("clip", "")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--enroll", nargs=2, type=float, metavar=("START", "END"))
    args = ap.parse_args()
    res = run(args.video, tuple(args.enroll) if args.enroll else None,
              lambda i, n: print(f"  {i}/{n} frames", end="\r"))
    name = os.path.splitext(os.path.basename(args.video))[0]
    out_dir = os.path.join(OUT_DIR, name)
    events = export_clips(args.video, res["events"], os.path.join(out_dir, "clips"))
    write_events_csv(events, os.path.join(out_dir, "events.csv"))
    print(f"enrolled face from {res['enroll_window'][0]:.1f}-{res['enroll_window'][1]:.1f}s; "
          f"subject tracked in {res['subject_frames_pct']:.1f}% of frames")
    for k, e in enumerate(events, 1):
        print(f"{k}. {classifier.fmt_time(e['start'])}-{classifier.fmt_time(e['end'])} {e['behavior']} "
              f"({e['confidence']:.2f}) -> {e['clip']}")
    print(f"events.csv + clips in {out_dir}")


if __name__ == "__main__":
    main()
