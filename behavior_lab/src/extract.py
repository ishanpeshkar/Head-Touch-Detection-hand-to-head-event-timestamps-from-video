"""Stage 1: run face / pose / hand models over a video once and cache the raw results.

Everything downstream (subject selection, features, classifier) reads this cache, so the slow
model pass runs once per video.

Usage:
    python behavior_lab/src/extract.py data/new/test_video.mp4
"""

import argparse
import hashlib
import os
import pickle
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

LAB_DIR = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR = os.path.join(LAB_DIR, "models")
CACHE_DIR = os.path.join(LAB_DIR, "outputs", "cache")


def _content_hash(video_path: str, chunk: int = 2**20) -> str:
    """Hash of the file's bytes, so the cache is keyed by content, not by path or mtime.

    Re-uploading the same video (a new copy, with a fresh mtime) then reuses the same cache
    instead of re-running the several-minutes-long model pass.
    """
    h = hashlib.sha1()
    with open(video_path, "rb") as f:
        while chunk_bytes := f.read(chunk):
            h.update(chunk_bytes)
    return h.hexdigest()[:20]


def cache_path(video_path: str) -> str:
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(CACHE_DIR, f"{stem}_{_content_hash(video_path)}.pkl")


def _landmarkers():
    hands = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=os.path.join(MODELS_DIR, "hand_landmarker.task")),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=4,
            min_hand_detection_confidence=0.4,
            min_tracking_confidence=0.4,
        )
    )
    poses = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=os.path.join(MODELS_DIR, "pose_landmarker_full.task")),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=2,
            min_pose_detection_confidence=0.4,
            min_tracking_confidence=0.4,
        )
    )
    return hands, poses


def make_face_models(width: int, height: int):
    det = cv2.FaceDetectorYN.create(
        os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx"), "", (width, height), score_threshold=0.6
    )
    rec = cv2.FaceRecognizerSF.create(os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx"), "")
    return det, rec


def detect_faces(det, rec, frame):
    """Return (faces (k,15) [x,y,w,h, 5 landmark xy pairs, score], embeddings (k,128))."""
    _, faces = det.detect(frame)
    if faces is None or len(faces) == 0:
        return np.zeros((0, 15), np.float32), np.zeros((0, 128), np.float32)
    embs = [rec.feature(rec.alignCrop(frame, f)).reshape(-1) for f in faces]
    return faces.astype(np.float32), np.array(embs, np.float32)


def extract_video(video_path: str, progress=None) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    hands, poses = _landmarkers()
    det, rec = make_face_models(w, h)

    frames = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t_ms = int(round(idx * 1000.0 / fps))
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pr = poses.detect_for_video(mp_img, t_ms)
        hr = hands.detect_for_video(mp_img, t_ms)
        faces, embs = detect_faces(det, rec, frame)
        frames.append(
            {
                "faces": faces,
                "embs": embs,
                "poses": [np.array([[l.x, l.y, l.visibility] for l in p], np.float32) for p in pr.pose_landmarks],
                "hands": [np.array([[l.x, l.y, l.z] for l in hand], np.float32) for hand in hr.hand_landmarks],
            }
        )
        idx += 1
        if progress and idx % 50 == 0:
            progress(idx, total)
    cap.release()
    return {"fps": fps, "width": w, "height": h, "frames": frames}


def load_or_extract(video_path: str, progress=None) -> dict:
    path = cache_path(video_path)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    data = extract_video(video_path, progress)
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)
    return data


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    args = ap.parse_args()
    t0 = time.time()
    d = load_or_extract(args.video, lambda i, n: print(f"  {i}/{n} frames", end="\r"))
    print(f"\n{len(d['frames'])} frames cached in {time.time() - t0:.0f}s -> {cache_path(args.video)}")
