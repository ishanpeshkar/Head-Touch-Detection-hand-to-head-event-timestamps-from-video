"""Stage 2: learn the subject's face from the video, then pick the subject in every frame.

Enrollment: average the SFace embeddings of the (single) face seen in an enrollment window.
Focus: in each frame, the detected face most similar to the enrolled identity (cosine >= threshold)
is the subject. Everyone else in the frame is ignored downstream.
"""

import numpy as np

# SFace's recommended "same person" cosine threshold is 0.363, but on this footage a relative's face
# scored up to ~0.51 against the subject while the subject's own median was ~0.9, so the strong
# threshold is set well above that; the weak one only applies together with position tracking.
STRONG_MATCH = 0.55
WEAK_MATCH = 0.25


def _unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def auto_enrollment_window(data, seconds: float = 8.0):
    """Pick the calmest stretch where exactly one confident face is visible: (start_sec, end_sec).

    Used when the user gives no window. 'Calm' = lowest face-center movement.
    """
    fps = data["fps"]
    n = len(data["frames"])
    win = int(seconds * fps)
    centers = np.full((n, 2), np.nan)
    single = np.zeros(n, bool)
    for i, fr in enumerate(data["frames"]):
        if len(fr["faces"]) == 1 and fr["faces"][0][-1] > 0.85:
            f = fr["faces"][0]
            centers[i] = (f[0] + f[2] / 2, f[1] + f[3] / 2)
            single[i] = True
    best, best_score = 0, np.inf
    for s in range(0, max(1, n - win), int(fps)):
        seg = slice(s, s + win)
        if single[seg].mean() < 0.95:
            continue
        score = np.nanstd(centers[seg], axis=0).sum()
        if score < best_score:
            best, best_score = s, score
    return best / fps, (best + win) / fps


def enroll(data, start_sec: float, end_sec: float) -> np.ndarray:
    """Mean unit embedding of the single face seen in [start_sec, end_sec]."""
    fps = data["fps"]
    embs = []
    for fr in data["frames"][int(start_sec * fps) : int(end_sec * fps)]:
        if len(fr["faces"]) == 1 and fr["faces"][0][-1] > 0.8:
            embs.append(fr["embs"][0])
    if len(embs) < 10:
        raise ValueError(f"Enrollment window {start_sec:.1f}-{end_sec:.1f}s has too few clear single-face frames ({len(embs)}).")
    return _unit(_unit(np.array(embs)).mean(axis=0))


def find_subject(data, identity: np.ndarray, strong: float = STRONG_MATCH, weak: float = WEAK_MATCH,
                 track_seconds: float = 1.0):
    """Per frame: (face_row (15,) or None, similarity float).

    A face is the subject if it matches strongly, or matches weakly (head turned, blur, partly
    covered by a hand) AND sits where the subject was within the last `track_seconds`.
    """
    fps = data["fps"]
    last_box, last_i = None, -10**9
    out = []
    for i, fr in enumerate(data["frames"]):
        if len(fr["faces"]) == 0:
            out.append((None, 0.0))
            continue
        sims = _unit(fr["embs"]) @ identity
        j = int(np.argmax(sims))
        face, sim = fr["faces"][j], float(sims[j])
        ok = sim >= strong
        if not ok and sim >= weak and last_box is not None and (i - last_i) <= track_seconds * fps:
            cx, cy = face[0] + face[2] / 2, face[1] + face[3] / 2
            lx, ly = last_box[0] + last_box[2] / 2, last_box[1] + last_box[3] / 2
            ok = abs(cx - lx) < last_box[2] and abs(cy - ly) < last_box[3]
        if ok:
            last_box, last_i = face, i
            out.append((face, sim))
        else:
            out.append((None, sim))
    return out
