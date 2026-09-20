"""Re-encode a video to H.264 so it plays in browsers and previews on GitHub.

`live_demo.py --record` writes with OpenCV, which can only produce MPEG-4 Part 2
('mp4v') unless an H.264 encoder library is installed; browsers generally cannot
play that. This converts the recording using the ffmpeg binary that ships with the
`imageio-ffmpeg` package (a development-only helper, not in requirements.txt):

    pip install imageio-ffmpeg
    python src/reencode_h264.py outputs/demo_raw_1.mp4 demo/demo_1.mp4
"""

import argparse
import subprocess

import imageio_ffmpeg


def main():
    ap = argparse.ArgumentParser(description="Re-encode a video to browser-playable H.264.")
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--crf", type=int, default=25, help="Quality: lower = better/larger (default 25)")
    args = ap.parse_args()

    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", args.src,
           "-c:v", "libx264", "-preset", "slow", "-crf", str(args.crf),
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", args.dst]
    subprocess.run(cmd, check=True)
    print(f"wrote {args.dst}")


if __name__ == "__main__":
    main()
