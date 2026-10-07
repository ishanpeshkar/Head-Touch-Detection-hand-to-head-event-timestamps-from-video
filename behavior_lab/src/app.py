"""Upload UI: upload a video, learn the person's face from it, detect behaviors, get a clip per event.

Run from the behavior_lab folder:
    .venv/Scripts/streamlit run src/app.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st

import classifier
import detect
import extract
import face_focus

st.set_page_config(page_title="Behavior log", layout="wide")
st.title("Behavior log")
st.caption("Upload a recording. The person is learned from the video's face, then behaviors are logged "
           "with timestamps and saved as clips. Everyone else in the frame is ignored.")

UPLOAD_DIR = os.path.join(extract.LAB_DIR, "outputs", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

up = st.file_uploader("Video", type=["mp4", "mov", "avi", "mkv"])
if up is None:
    st.info("Upload a video to begin.")
    st.stop()

video_path = os.path.join(UPLOAD_DIR, os.path.basename(up.name))
if not os.path.exists(video_path) or os.path.getsize(video_path) != up.size:
    with open(video_path, "wb") as f:
        f.write(up.getbuffer())

with st.expander("Video preview", expanded=False):
    st.video(video_path)

# 1. face / pose / hand extraction (slow, cached per video by content hash - a re-upload of the
# same video, even under a new filename or mtime, reuses the cache instead of re-analysing)
cache_ready = os.path.exists(extract.cache_path(video_path))
if not cache_ready:
    st.warning("First run on this video: analysing every frame. This takes a few minutes for a short video "
               "and is only done once.")
bar = st.progress(0.0, text="Analysing video")
data = extract.load_or_extract(video_path, lambda i, n: bar.progress(min(i / n, 1.0), text=f"Analysing frame {i}/{n}"))
bar.empty()
duration = len(data["frames"]) / data["fps"]

# 2. who is the person? (learned from the video)
st.subheader("1. Who to follow")
auto = face_focus.auto_enrollment_window(data)
manual = st.checkbox("Choose the enrollment stretch myself", value=False)
window = auto
if manual:
    window = st.slider("Seconds where only the person is visible, looking at the camera",
                       0.0, float(duration), (float(auto[0]), float(auto[1])), step=0.5)
try:
    thumb = detect.enrollment_thumbnail(video_path, data, window)
    c1, c2 = st.columns([1, 4])
    if thumb is not None:
        c1.image(thumb, caption="Face learned")
    c2.write(f"Learned from {window[0]:.1f}s to {window[1]:.1f}s. If this is not the right person, "
             "tick the box above and pick a stretch where only they are visible.")
except ValueError as e:
    st.error(str(e))
    st.stop()

conf = st.slider("Minimum confidence to log a behavior", 0.3, 0.9, 0.5, 0.05)

# 3. detect + clip
if st.button("Detect behaviors", type="primary"):
    with st.spinner("Detecting and cutting clips"):
        res = detect.run(video_path, window, min_conf=conf)
        out_dir = os.path.join(detect.OUT_DIR, os.path.splitext(os.path.basename(video_path))[0])
        events = detect.export_clips(video_path, res["events"], os.path.join(out_dir, "clips"))
        csv_path = os.path.join(out_dir, "events.csv")
        detect.write_events_csv(events, csv_path)
    st.session_state["result"] = (events, csv_path, res["subject_frames_pct"])

if "result" in st.session_state:
    events, csv_path, pct = st.session_state["result"]
    st.subheader("2. Behavior log")
    st.caption(f"Person found in {pct:.0f}% of frames. Log describes behavior only; nothing is scored or ranked.")
    if not events:
        st.write("No behaviors detected.")
    st.dataframe(
        [{"#": k, "start": classifier.fmt_time(e["start"]), "end": classifier.fmt_time(e["end"]),
          "behavior": e["behavior"].replace("_", " "), "confidence": round(e["confidence"], 2)}
         for k, e in enumerate(events, 1)],
        hide_index=True, use_container_width=True)
    with open(csv_path, "rb") as f:
        st.download_button("Download log (CSV)", f, file_name="behavior_log.csv")
    st.subheader("3. Clips")
    cols = st.columns(3)
    for k, e in enumerate(events):
        with cols[k % 3]:
            st.markdown(f"**{k + 1}. {e['behavior'].replace('_', ' ')}** · {classifier.fmt_time(e['start'])}")
            st.video(e["clip"])
