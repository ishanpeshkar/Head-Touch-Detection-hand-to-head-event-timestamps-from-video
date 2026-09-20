"""Test whether a different head shape fixes the quick-touch miss, without
re-running the landmarkers.

Hypothesis: the original head model (a circle centred at eye level with radius =
full ear-to-ear width) is too large and too low. Open hands raised beside the
face fall inside it, and it can't tell them from a hand on top of the head. A
smaller circle centred higher might separate the two, which would let the speed
gate be loosened enough to catch a quick touch without flooding the output with
false positives.

Protocol (fixed before running, to limit tuning to the answer):
  * Candidate family only:  up in {0, .2, .4, .6}  x  radius in {.5, .6, .7, .8, .9, 1.0}
    x  speed limit in {0.5, 1.0, 1.5, none}.  Distance threshold 1.0 (radius absorbs
    scale), enter 3 frames, exit 10 frames -- unchanged from the current detector.
  * Selection score: F1 under the overlap matching rule; ties -> fewer detections.
  * Baseline: up=0, radius=1.0, speed limit 0.5 (the current defaults).
  * In-sample numbers (best on all 5 touches) are reported, but the honest estimate
    is leave-one-out: for each touch, choose settings using the other four, then
    check whether the held-out touch is found. Detections that match the held-out
    touch are ignored while selecting, so finding it is not penalised as a false
    positive.

Usage:
    python src/geometry_experiment.py outputs/distance_signal_geometry.csv \\
        annotations/test_video_task_annotations.csv
"""

import argparse
import itertools

from evaluate import load_events, match_events
from head_touch_detector import apply_head_geometry, extract_events, load_signal_csv

UPS = [0.0, 0.2, 0.4, 0.6]
RADII = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SPEEDS = [0.5, 1.0, 1.5, None]
BASELINE = (0.0, 1.0, 0.5)


def detect(rows, up, radius, speed):
    apply_head_geometry(rows, up, radius)
    events = extract_events(rows, 1.0, 3, 10, velocity_threshold=speed)
    return [dict(event_id=i, start_time=e["start_time"], contact_time=e["contact_time"],
                 end_time=e["end_time"], hand=e["hand"], notes="")
            for i, e in enumerate(events, start=1)]


def score(gt, det, mode="overlap", ignore_touch=None, full_gt=None):
    """F1 etc. against `gt`. If ignore_touch is given, detections that match that
    (held-out) touch under `full_gt` are dropped first."""
    if ignore_touch is not None:
        matches, _, _ = match_events(full_gt, det, 0.5, mode)
        drop = {id(d) for g, d, _ in matches if g is ignore_touch}
        det = [d for d in det if id(d) not in drop]
    m, ug, ud = match_events(gt, det, 0.5, mode)
    tp, fn, fp = len(m), len(ug), len(ud)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return dict(f1=f1, p=p, r=r, tp=tp, fp=fp, fn=fn, n=len(det),
                missed=[g["event_id"] for g in ug])


def fmt(params):
    up, rad, spd = params
    return f"up={up} radius={rad} speed={spd}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("signal_csv")
    ap.add_argument("ground_truth_csv")
    args = ap.parse_args()

    rows = load_signal_csv(args.signal_csv)
    gt = load_events(args.ground_truth_csv)
    grid = list(itertools.product(UPS, RADII, SPEEDS))

    # Sanity: the original head model must reproduce the stored distances.
    stored = [r["normalized_distance"] for r in rows]
    apply_head_geometry(rows, 0.0, 1.0)
    worst = max(abs(a - r["normalized_distance"]) for a, r in zip(stored, rows))
    print(f"sanity: recomputed original-geometry distances differ from stored by at most {worst:.4f}\n")

    results = {}
    for params in grid:
        det = detect(rows, *params)
        results[params] = (det, {mode: score(gt, det, mode) for mode in ("overlap", "contact")})

    print("Baseline (current defaults):", fmt(BASELINE))
    for mode in ("contact", "overlap"):
        s = results[BASELINE][1][mode]
        print(f"  {mode:8s} F1={s['f1']:.2f} P={s['p']:.2f} R={s['r']:.2f} TP={s['tp']} FP={s['fp']} missed={s['missed']}")

    ranked = sorted(grid, key=lambda g: (-results[g][1]["overlap"]["f1"], results[g][1]["overlap"]["n"]))
    print("\nTop 8 on all 5 touches (in-sample, overlap-rule F1):")
    for g in ranked[:8]:
        s, sc = results[g][1]["overlap"], results[g][1]["contact"]
        print(f"  {fmt(g):38s} overlap F1={s['f1']:.2f} P={s['p']:.2f} R={s['r']:.2f} TP={s['tp']} FP={s['fp']} "
              f"missed={s['missed']} | strict F1={sc['f1']:.2f} R={sc['r']:.2f}")

    for spd in (0.5, 1.5):
        print(f"\nTP/FP grid, overlap rule, speed limit {spd} (rows: up, columns: radius {RADII})")
        for up in UPS:
            cells = []
            for rad in RADII:
                s = results[(up, rad, spd)][1]["overlap"]
                cells.append(f"{s['tp']}/{s['fp']:<2d}")
            print(f"  up={up}: " + "  ".join(cells))

    print("\nLeave-one-out (select on 4 touches, test on the held-out one):")
    found = 0
    for held in gt:
        train = [g for g in gt if g is not held]
        best = None
        for g in grid:
            det = results[g][0]
            sc = score(train, det, "overlap", ignore_touch=held, full_gt=gt)
            key = (-sc["f1"], sc["n"])
            if best is None or key < best[0]:
                best = (key, g)
        g = best[1]
        m, _, _ = match_events(gt, results[g][0], 0.5, "overlap")
        hit = any(x[0] is held for x in m)
        found += hit
        s = results[g][1]["overlap"]
        print(f"  hold out GT#{held['event_id']}: chose {fmt(g):36s} -> held-out found={hit}  "
              f"(all-5 overlap: FP={s['fp']} P={s['p']:.2f})")
    print(f"  leave-one-out recall: {found}/{len(gt)} held-out touches found "
          f"(baseline finds {results[BASELINE][1]['overlap']['tp']}/{len(gt)} with fixed defaults)")


if __name__ == "__main__":
    main()
