"""Sweep detection parameters against a cached distance signal to pick a
threshold/debounce combo, without re-running the (slow) landmark
extraction each time.

This is what actually produced the `--threshold` / `--enter-frames` /
`--exit-frames` defaults documented in the README: not a guess, but a
grid search against the 3 ground-truth events, scored by F1 (ties broken
by lower mean timing error). With only 3 ground-truth events this is a
coarse signal and shouldn't be over-trusted -- see the README's Results
section for the caveat -- but it's more principled than hand-picking
numbers.

Usage:
    python src/tune_threshold.py outputs/distance_signal.csv \\
        annotations/test_video_task_annotations.csv --tolerance 0.5
"""

import argparse

from head_touch_detector import extract_events, load_signal_csv
from evaluate import load_events, match_events


def main():
    parser = argparse.ArgumentParser(description="Grid-search detection thresholds against ground truth.")
    parser.add_argument("signal_csv")
    parser.add_argument("ground_truth_csv")
    parser.add_argument("--tolerance", type=float, default=0.5)
    args = parser.parse_args()

    signal_rows = load_signal_csv(args.signal_csv)
    ground_truth = load_events(args.ground_truth_csv)

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    enter_options = [2, 3, 4, 5]
    exit_options = [3, 5, 8, 10]
    velocity_options = [None, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]  # head-widths/sec; None = gate off

    results = []
    for threshold in thresholds:
        for enter_frames in enter_options:
            for exit_frames in exit_options:
                for velocity_threshold in velocity_options:
                    events = extract_events(signal_rows, threshold, enter_frames, exit_frames,
                                             velocity_threshold=velocity_threshold)
                    detected = [{
                        "event_id": i, "start_time": e["start_time"], "contact_time": e["contact_time"],
                        "end_time": e["end_time"], "hand": e["hand"], "notes": "",
                    } for i, e in enumerate(events, start=1)]
                    matches, unmatched_gt, unmatched_det = match_events(ground_truth, detected, args.tolerance)
                    tp, fn, fp = len(matches), len(unmatched_gt), len(unmatched_det)
                    precision = tp / (tp + fp) if (tp + fp) else 0.0
                    recall = tp / (tp + fn) if (tp + fn) else 0.0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
                    mae = sum(m[2] for m in matches) / tp if tp else float("inf")
                    results.append((f1, -mae, precision, recall, tp, fp, fn,
                                    threshold, enter_frames, exit_frames, velocity_threshold))

    results.sort(key=lambda r: (r[0], r[1]), reverse=True)

    print(f"{'F1':>5} {'prec':>5} {'rec':>5} {'TP':>3} {'FP':>4} {'FN':>3} {'MAE':>6}  "
          f"{'thresh':>6} {'enter':>5} {'exit':>4} {'vel':>5}")
    for (f1, neg_mae, precision, recall, tp, fp, fn,
         threshold, enter_frames, exit_frames, velocity_threshold) in results[:20]:
        mae = -neg_mae
        mae_str = f"{mae:.2f}" if mae != float("inf") else "  n/a"
        vel_str = "off" if velocity_threshold is None else f"{velocity_threshold:.2f}"
        print(f"{f1:5.2f} {precision:5.2f} {recall:5.2f} {tp:3d} {fp:4d} {fn:3d} {mae_str:>6}  "
              f"{threshold:6.2f} {enter_frames:5d} {exit_frames:4d} {vel_str:>5}")


if __name__ == "__main__":
    main()
