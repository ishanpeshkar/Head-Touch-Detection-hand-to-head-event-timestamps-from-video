"""Phase 1: inspect a video file and report its basic properties.

Usage:
    python src/inspect_video.py data/test_video.mp4
    python src/inspect_video.py data/test_video.mp4 --sample-frames 5 --out outputs/samples
"""

import argparse
import os

import cv2

from timeutils import format_timestamp


def fourcc_to_str(fourcc_int: float) -> str:
    fourcc_int = int(fourcc_int)
    chars = [chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)]
    return "".join(chars).strip()


def inspect_video(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cap.get(cv2.CAP_PROP_FOURCC)
    duration = frame_count / fps if fps else 0.0

    cap.release()

    return {
        "filename": os.path.basename(video_path),
        "path": os.path.abspath(video_path),
        "size_mb": round(os.path.getsize(video_path) / (1024 * 1024), 2),
        "fps": round(fps, 3),
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_sec": round(duration, 3),
        "duration_hms": format_timestamp(duration),
        "codec": fourcc_to_str(fourcc),
    }


def sample_frames(video_path: str, n: int, out_dir: str) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0 or n <= 0:
        cap.release()
        return

    os.makedirs(out_dir, exist_ok=True)
    indices = [int(i * (frame_count - 1) / max(n - 1, 1)) for i in range(n)]

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        out_path = os.path.join(out_dir, f"frame_{idx:06d}.jpg")
        cv2.imwrite(out_path, frame)

    cap.release()


def main():
    parser = argparse.ArgumentParser(description="Inspect a video file's basic properties.")
    parser.add_argument("video", help="Path to the input video file")
    parser.add_argument("--sample-frames", type=int, default=0,
                         help="Number of evenly-spaced frames to save as JPEGs")
    parser.add_argument("--out", default="outputs/samples",
                         help="Directory to save sampled frames into")
    args = parser.parse_args()

    info = inspect_video(args.video)
    print("Video properties:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    if args.sample_frames > 0:
        sample_frames(args.video, args.sample_frames, args.out)
        print(f"\nSaved {args.sample_frames} sample frame(s) to {args.out}")


if __name__ == "__main__":
    main()
