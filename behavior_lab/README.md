# behavior_lab: learned behavior detection (ear cover, hair twirl, head banging, head touch)

Separate from the rule-based head-touch detector in `../src`. Nothing here imports from, writes to,
or changes that project. It has its own virtual environment, models, annotations and outputs.

## Setup
```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```
Model files go in `models/` (gitignored): `hand_landmarker.task`, `pose_landmarker_full.task`
(MediaPipe) and `face_detection_yunet_2023mar.onnx`, `face_recognition_sface_2021dec.onnx` (OpenCV zoo).

## How it works
1. `extract.py`   faces (+identity embeddings), body pose and hands for every frame, cached once per video.
2. `face_focus.py` learns the person's face from a stretch of the video where only they are visible, then
   picks that person in every frame. Others are ignored.
3. `features.py`  subject-only geometry (hand-to-ear, hand-above-head, tapping rate, head motion...) in windows.
4. `classifier.py` random forest trained on windows labelled from an annotation CSV; time-blocked cross-validation.
5. `detect.py`    trained model on a video -> events.csv + one H.264 clip per event.
6. `app.py`       upload UI (Streamlit).

## Use
```
# train + evaluate on an annotated video (writes outputs/model/behavior_model.pkl)
.venv\Scripts\python src/classifier.py ..\data\new\test_video.mp4 annotations/new_video_ground_truth.csv
# detect on any video
.venv\Scripts\python src/detect.py some_video.mp4
# upload UI
.venv\Scripts\streamlit run src/app.py
```
