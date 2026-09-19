"""Phase 4: evaluate detected events against manually-annotated ground truth.

Matches detected events to ground-truth events by proximity of
`contact_time` (greedy nearest-neighbor, one-to-one) within a tolerance
window, then reports precision / recall / F1 and per-event timing error.

Hand label (left/right) is reported as a diagnostic, not used to gate a
match -- MediaPipe's handedness convention can come out mirrored relative
to the real-world hand depending on how the source video was captured, so
we don't want a label mismatch to hide an otherwise-correct time match.

Usage:
    python src/evaluate.py annotations/test_video_task_annotations.csv \\
        outputs/detected_events.csv --tolerance 0.5
"""

import argparse
import csv

from timeutils import parse_timestamp, format_timestamp


def load_events(path: str):
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        events = []
        for row in reader:
            events.append({
                "event_id": row["event_id"],
                "start_time": parse_timestamp(row["start_time"]),
                "contact_time": parse_timestamp(row["contact_time"]),
                "end_time": parse_timestamp(row["end_time"]),
                "hand": row["hand"],
                "notes": row.get("notes", ""),
            })
    return events


def match_events(ground_truth, detected, tolerance: float):
    """Greedy nearest-neighbor one-to-one matching on contact_time.

    Returns (matches, unmatched_gt, unmatched_det) where matches is a list
    of (gt_event, det_event, abs_time_diff) sorted by gt contact_time.
    """
    candidates = []
    for gt in ground_truth:
        for det in detected:
            diff = abs(gt["contact_time"] - det["contact_time"])
            if diff <= tolerance:
                candidates.append((diff, gt, det))
    candidates.sort(key=lambda c: c[0])

    matched_gt_ids = set()
    matched_det_ids = set()
    matches = []
    for diff, gt, det in candidates:
        if id(gt) in matched_gt_ids or id(det) in matched_det_ids:
            continue
        matched_gt_ids.add(id(gt))
        matched_det_ids.add(id(det))
        matches.append((gt, det, diff))

    matches.sort(key=lambda m: m[0]["contact_time"])
    unmatched_gt = [gt for gt in ground_truth if id(gt) not in matched_gt_ids]
    unmatched_det = [det for det in detected if id(det) not in matched_det_ids]
    return matches, unmatched_gt, unmatched_det


def report(ground_truth, detected, tolerance: float):
    matches, unmatched_gt, unmatched_det = match_events(ground_truth, detected, tolerance)

    tp = len(matches)
    fn = len(unmatched_gt)
    fp = len(unmatched_det)
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")
    mean_abs_error = sum(m[2] for m in matches) / tp if tp else float("nan")

    print(f"Ground truth events: {len(ground_truth)}   Detected events: {len(detected)}   "
          f"Tolerance: +/-{tolerance:.2f}s\n")

    print("Matched (TP):")
    if matches:
        for gt, det, diff in matches:
            hand_flag = "" if gt["hand"].lower() == det["hand"].lower() else "  [hand label mismatch]"
            print(f"  GT#{gt['event_id']} contact={format_timestamp(gt['contact_time'])} ({gt['hand']})  <->  "
                  f"det contact={format_timestamp(det['contact_time'])} ({det['hand']})  "
                  f"diff={diff:+.2f}s{hand_flag}")
    else:
        print("  (none)")

    print("\nMissed (False Negatives):")
    if unmatched_gt:
        for gt in unmatched_gt:
            print(f"  GT#{gt['event_id']} contact={format_timestamp(gt['contact_time'])} ({gt['hand']})  "
                  f"notes={gt['notes']!r}")
    else:
        print("  (none)")

    print("\nExtra detections (False Positives):")
    if unmatched_det:
        for det in unmatched_det:
            print(f"  det#{det['event_id']} contact={format_timestamp(det['contact_time'])} ({det['hand']})  "
                  f"notes={det['notes']!r}")
    else:
        print("  (none)")

    print(f"\nPrecision: {precision:.2f}   Recall: {recall:.2f}   F1: {f1:.2f}   "
          f"Mean |timing error| on matches: {mean_abs_error:.2f}s")

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "mean_abs_error": mean_abs_error,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate detected events against ground truth.")
    parser.add_argument("ground_truth_csv")
    parser.add_argument("detected_csv")
    parser.add_argument("--tolerance", type=float, default=0.5,
                         help="Seconds of allowed difference between GT and detected contact_time")
    args = parser.parse_args()

    ground_truth = load_events(args.ground_truth_csv)
    detected = load_events(args.detected_csv)
    report(ground_truth, detected, args.tolerance)


if __name__ == "__main__":
    main()
