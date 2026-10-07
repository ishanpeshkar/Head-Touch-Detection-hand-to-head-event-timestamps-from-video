"""Stage 4: learn behaviors from labelled windows, predict timed events.

Usage (train + honest evaluation on one annotated video):
    python behavior_lab/src/classifier.py data/new/test_video.mp4 behavior_lab/annotations/new_video_ground_truth.csv
"""

import argparse
import csv
import os
import pickle
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier

import extract
import face_focus
import features

CLASSES = ["none", "ear_cover", "hair_twirl", "head_touch", "head_bang"]
MODEL_DIR = os.path.join(extract.LAB_DIR, "outputs", "model")


def parse_time(s: str) -> float:
    m, sec = s.split(":")
    return int(m) * 60 + float(sec)


def fmt_time(t: float) -> str:
    return f"{int(t // 60):02d}:{t % 60:05.2f}"


def load_ground_truth(path):
    with open(path, newline="") as f:
        return [
            {"start": parse_time(r["start_time"]), "end": parse_time(r["end_time"]), "behavior": r["behavior"],
             "person": r.get("person", "subject")}
            for r in csv.DictReader(f)
            if r["person"] == "subject" and r["behavior"] in CLASSES
        ]


def label_windows(starts, win_sec, events, min_overlap=0.5):
    y = np.zeros(len(starts), int)
    for k, s in enumerate(starts):
        for ev in events:
            ov = min(s + win_sec, ev["end"]) - max(s, ev["start"])
            if ov >= min_overlap * min(win_sec, ev["end"] - ev["start"]):
                y[k] = CLASSES.index(ev["behavior"])
    return y


def make_model():
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
                                  random_state=0, n_jobs=-1)


def prepare(video_path, enroll_window=None):
    data = extract.load_or_extract(video_path)
    if enroll_window is None:
        enroll_window = face_focus.auto_enrollment_window(data)
    identity = face_focus.enroll(data, *enroll_window)
    subject = face_focus.find_subject(data, identity)
    sig = features.frame_signals(data, subject)
    X, starts, names = features.window_features(sig, data["fps"])
    return data, identity, subject, X, starts, names, enroll_window


def windows_to_events(starts, win_sec, proba, min_conf=0.5, min_len=0.75, merge_gap=1.0):
    """Merge consecutive same-class windows into (start, end, behavior, confidence) events."""
    pred = proba.argmax(1)
    conf = proba.max(1)
    events, cur = [], None
    for s, p, c in zip(starts, pred, conf):
        if p != 0 and c >= min_conf:
            if cur and cur["cls"] == p and s <= cur["end"] + merge_gap:
                cur["end"] = s + win_sec
                cur["conf"].append(c)
            else:
                if cur:
                    events.append(cur)
                cur = {"cls": p, "start": s, "end": s + win_sec, "conf": [c]}
    if cur:
        events.append(cur)
    return [
        {"start": e["start"], "end": e["end"], "behavior": CLASSES[e["cls"]], "confidence": float(np.mean(e["conf"]))}
        for e in events
        if e["end"] - e["start"] >= min_len
    ]


def match(events, gt):
    """Greedy overlap match. Returns (matches, false_positives, missed)."""
    used, matches, fps_ = set(), [], []
    for e in events:
        best, bo = None, 0.0
        for k, g in enumerate(gt):
            ov = min(e["end"], g["end"]) - max(e["start"], g["start"])
            if k not in used and ov > bo:
                best, bo = k, ov
        if best is not None and bo > 0:
            used.add(best)
            matches.append((e, gt[best]))
        else:
            fps_.append(e)
    return matches, fps_, [g for k, g in enumerate(gt) if k not in used]


def cross_validate(X, y, starts, win_sec, gt, n_folds=6, purge=2.0):
    """Contiguous time-block CV with a purge gap, so neighbouring (overlapping) windows never leak."""
    edges = np.linspace(0, starts.max() + win_sec, n_folds + 1)
    proba = np.zeros((len(y), len(CLASSES)))
    for f in range(n_folds):
        lo, hi = edges[f], edges[f + 1]
        test = (starts >= lo) & (starts < hi)
        train = (starts + win_sec <= lo - purge) | (starts >= hi + purge)
        m = make_model().fit(X[train], y[train])
        p = m.predict_proba(X[test])
        for j, c in enumerate(m.classes_):
            proba[np.where(test)[0], c] = p[:, j]
    return proba


def report(events, gt, title):
    matches, false_pos, missed = match(events, gt)
    correct = [m for m in matches if m[0]["behavior"] == m[1]["behavior"]]
    print(f"\n== {title} ==")
    print(f"ground-truth events {len(gt)} | detected {len(events)} | found (any label) {len(matches)} | "
          f"found with correct label {len(correct)} | false alarms {len(false_pos)} | missed {len(missed)}")
    for e, g in matches:
        ok = "OK " if e["behavior"] == g["behavior"] else "MIS"
        print(f"  {ok} truth {fmt_time(g['start'])}-{fmt_time(g['end'])} {g['behavior']:<10} "
              f"-> pred {fmt_time(e['start'])}-{fmt_time(e['end'])} {e['behavior']:<10} conf {e['confidence']:.2f}")
    for e in false_pos:
        print(f"  FA  pred {fmt_time(e['start'])}-{fmt_time(e['end'])} {e['behavior']} conf {e['confidence']:.2f}")
    for g in missed:
        print(f"  MISSED truth {fmt_time(g['start'])}-{fmt_time(g['end'])} {g['behavior']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("ground_truth")
    ap.add_argument("--enroll", nargs=2, type=float, metavar=("START", "END"), default=None)
    args = ap.parse_args()

    data, identity, subject, X, starts, names, ew = prepare(args.video, args.enroll)
    win_sec = 1.5
    gt = load_ground_truth(args.ground_truth)
    y = label_windows(starts, win_sec, gt)
    print(f"enrollment window {ew[0]:.1f}-{ew[1]:.1f}s | windows {len(y)} | per class "
          + ", ".join(f"{c}={int((y == i).sum())}" for i, c in enumerate(CLASSES)))

    proba = cross_validate(X, y, starts, win_sec, gt)
    report(windows_to_events(starts, win_sec, proba), gt, "held-out (time-blocked cross-validation)")

    model = make_model().fit(X, y)
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "behavior_model.pkl"), "wb") as f:
        pickle.dump({"model": model, "names": names, "win_sec": win_sec}, f)
    train_events = windows_to_events(starts, win_sec, model.predict_proba(X))
    report(train_events, gt, "on training data (optimistic - shown only for reference)")
    top = np.argsort(model.feature_importances_)[::-1][:8]
    print("\nmost useful features:", ", ".join(f"{names[i]} ({model.feature_importances_[i]:.2f})" for i in top))


if __name__ == "__main__":
    sys.exit(main())
