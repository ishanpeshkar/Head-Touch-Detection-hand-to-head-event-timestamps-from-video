"""Download the MediaPipe Tasks model bundles used by head_touch_detector.py.

They're gitignored (large binaries, ~13 MB total) -- run this once after
cloning / installing requirements.

Usage:
    python src/download_models.py
"""

import os
import urllib.request

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

MODELS = {
    "hand_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "pose_landmarker_lite.task":
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
}


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    for filename, url in MODELS.items():
        out_path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(out_path):
            print(f"already have {filename}, skipping")
            continue
        print(f"downloading {filename} ...")
        urllib.request.urlretrieve(url, out_path)
        print(f"  saved to {out_path}")


if __name__ == "__main__":
    main()
