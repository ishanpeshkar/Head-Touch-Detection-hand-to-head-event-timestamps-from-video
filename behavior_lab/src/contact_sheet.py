"""Make timestamped contact sheets of a video, to find behavior start/end times by eye.

Usage:
    python behavior_lab/src/contact_sheet.py data/new/test_video.mp4 --step 1.0 --start 0 --end 60
"""

import argparse
import os

import cv2
import numpy as np


def stamp(sec: float) -> str:
    return f"{int(sec // 60):02d}:{sec % 60:04.1f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--step", type=float, default=1.0, help="seconds between frames")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--rows", type=int, default=5)
    ap.add_argument("--out", default=os.path.join("behavior_lab", "outputs", "sheets"))
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    end = min(args.end or duration, duration)
    os.makedirs(args.out, exist_ok=True)

    times = np.arange(args.start, end, args.step)
    per_sheet = args.cols * args.rows
    tw, th = 320, 240
    for s in range(0, len(times), per_sheet):
        chunk = times[s : s + per_sheet]
        sheet = np.zeros((args.rows * th, args.cols * tw, 3), np.uint8)
        for i, t in enumerate(chunk):
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if not ok:
                continue
            tile = cv2.resize(frame, (tw, th))
            cv2.putText(tile, stamp(t), (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
            cv2.putText(tile, stamp(t), (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            r, c = divmod(i, args.cols)
            sheet[r * th : (r + 1) * th, c * tw : (c + 1) * tw] = tile
        path = os.path.join(args.out, f"sheet_{stamp(chunk[0]).replace(':', 'm')}.jpg")
        cv2.imwrite(path, sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
        print(path)


if __name__ == "__main__":
    main()
