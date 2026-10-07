"""Stage 3: per-frame and per-window features describing the SUBJECT only.

All geometry is expressed relative to the subject's face (origin = face centre, unit = face width),
so it does not depend on camera distance. Other people in the frame never enter the features.
"""

import numpy as np

# per-frame signal names, in column order
SIGNALS = [
    "n_hands",        # hands assigned to the subject
    "d_head",         # nearest hand landmark to the face centre
    "d_ear",          # nearest hand landmark to either ear
    "d_ear_both",     # max over the two ears of the nearest-landmark distance (small = both ears covered)
    "d_top",          # nearest hand landmark to the point above the head
    "hand_above",     # highest hand landmark above the face centre
    "in_head",        # fraction of hand landmarks inside the (enlarged) head box
    "hand_cy",        # hand centroid, vertical (relative)
    "hand_cx",        # hand centroid, horizontal (relative)
    "spread",         # mean distance of landmarks from the hand centroid (finger curl / fist)
    "wrist_d",        # nearest body-pose wrist to the face centre (works when hand landmarks fail)
    "wrist_dy",       # highest body-pose wrist above the face centre
    "elbow_up",       # highest body-pose elbow above the face centre
    "head_dx", "head_dy", "head_ds",   # face-centre motion and size change between frames
    "have_face",
]
FAR = 6.0  # value used for "no hand" distances
IDX = {n: i for i, n in enumerate(SIGNALS)}


def _pose_for(poses, face, W, H):
    """The pose whose nose is closest to the subject's face centre (or None)."""
    if not poses or face is None:
        return None
    c = np.array([face[0] + face[2] / 2, face[1] + face[3] / 2])
    best, bd = None, 1e9
    for p in poses:
        d = np.linalg.norm(np.array([p[0, 0] * W, p[0, 1] * H]) - c)
        if d < bd:
            best, bd = p, d
    return best if bd < 1.2 * face[2] else None



def frame_signals(data, subject):
    """(n_frames, len(SIGNALS)) float array. `subject` = output of face_focus.find_subject."""
    W, H = data["width"], data["height"]
    fps = data["fps"]
    n = len(data["frames"])
    out = np.zeros((n, len(SIGNALS)), np.float32)
    out[:, [IDX[k] for k in ("d_head", "d_ear", "d_ear_both", "d_top", "wrist_d")]] = FAR
    out[:, IDX["hand_above"]] = -FAR
    out[:, IDX["wrist_dy"]] = -FAR
    out[:, IDX["elbow_up"]] = -FAR

    last_face, last_i = None, -10**9
    prev = None
    for i, fr in enumerate(data["frames"]):
        face = subject[i][0]
        if face is not None:
            last_face, last_i = face, i
        elif last_face is not None and i - last_i <= 0.5 * fps:
            face = last_face  # short gap (hand covering the face): carry the box forward
        else:
            prev = None
            continue
        out[i, IDX["have_face"]] = 1.0 if subject[i][0] is not None else 0.5
        x, y, w, h = face[:4]
        c = np.array([x + w / 2, y + h / 2])
        s = w

        def rel(pts_norm):
            p = np.asarray(pts_norm)[:, :2] * [W, H]
            return (p - c) / s * [1, -1]  # y up

        ears = np.array([[-0.5, 0.0], [0.5, 0.0]])
        top = np.array([[0.0, 0.5 * h / w + 0.35]])

        # hands belonging to the subject: nearest face (of everyone in frame) must be the subject's
        mine = []
        for hand in fr["hands"]:
            wrist = hand[0, :2] * [W, H]
            faces = fr["faces"]
            dists = [np.linalg.norm(wrist - [f[0] + f[2] / 2, f[1] + f[3] / 2]) / f[2] for f in faces]
            d_me = np.linalg.norm(wrist - c) / s
            if d_me < 4.5 and all(d_me <= d for d in dists):
                mine.append(hand)
        out[i, IDX["n_hands"]] = len(mine)
        if mine:
            pts = np.concatenate([rel(hd) for hd in mine])
            out[i, IDX["d_head"]] = np.linalg.norm(pts, axis=1).min()
            dl = np.linalg.norm(pts - ears[0], axis=1).min()
            dr = np.linalg.norm(pts - ears[1], axis=1).min()
            out[i, IDX["d_ear"]] = min(dl, dr)
            out[i, IDX["d_ear_both"]] = max(dl, dr)
            out[i, IDX["d_top"]] = np.linalg.norm(pts - top[0], axis=1).min()
            out[i, IDX["hand_above"]] = pts[:, 1].max()
            box = (np.abs(pts[:, 0]) < 0.75) & (pts[:, 1] > -0.75) & (pts[:, 1] < 0.9 * h / w + 0.5)
            out[i, IDX["in_head"]] = box.mean()
            cen = pts.mean(axis=0)
            out[i, IDX["hand_cx"]], out[i, IDX["hand_cy"]] = cen
            out[i, IDX["spread"]] = np.linalg.norm(pts - cen, axis=1).mean()

        pose = _pose_for(fr["poses"], face, W, H)
        if pose is not None:
            wr = rel(pose[[15, 16]])
            el = rel(pose[[13, 14]])
            vis = pose[[15, 16], 2] > 0.3
            if vis.any():
                out[i, IDX["wrist_d"]] = np.linalg.norm(wr[vis], axis=1).min()
                out[i, IDX["wrist_dy"]] = wr[vis, 1].max()
            if (pose[[13, 14], 2] > 0.3).any():
                out[i, IDX["elbow_up"]] = el[pose[[13, 14], 2] > 0.3, 1].max()

        if prev is not None:
            out[i, IDX["head_dx"]] = (c[0] - prev[0]) / s
            out[i, IDX["head_dy"]] = (c[1] - prev[1]) / s
            out[i, IDX["head_ds"]] = (s - prev[2]) / s
        prev = (c[0], c[1], s)
    return out


def _reversals(v, min_amp):
    """Number of direction changes in v with swing larger than min_amp (tapping / oscillation)."""
    v = v[~np.isnan(v)]
    if len(v) < 3:
        return 0
    cnt, direction, ref = 0, 0, v[0]
    for x in v[1:]:
        if direction >= 0 and x < ref - min_amp:
            cnt += direction == 1
            direction, ref = -1, x
        elif direction <= 0 and x > ref + min_amp:
            cnt += direction == -1
            direction, ref = 1, x
        elif (direction >= 0 and x > ref) or (direction <= 0 and x < ref):
            ref = x
        if direction == 0:
            direction = 1 if x > v[0] else -1
    return cnt


def window_features(sig, fps, win_sec=1.5, stride_sec=0.25):
    """Slide a window over the per-frame signals. Returns (X, starts_sec, names)."""
    win, stride = int(win_sec * fps), int(stride_sec * fps)
    names = [f"{s}_{a}" for s in SIGNALS[:13] for a in ("mean", "min", "max", "std")]
    names += ["head_speed_mean", "head_speed_max", "head_dy_std", "contact_flips", "hand_y_flips", "wrist_d_flips",
              "hand_speed_mean", "hand_present", "face_present", "d_head_lt1", "d_ear_lt1"]
    rows, starts = [], []
    for a in range(0, len(sig) - win + 1, stride):
        w = sig[a : a + win]
        f = []
        for k in range(13):
            col = w[:, k]
            f += [col.mean(), col.min(), col.max(), col.std()]
        speed = np.hypot(w[:, IDX["head_dx"]], w[:, IDX["head_dy"]])
        dh = w[:, IDX["d_head"]]
        contact = (dh < 1.0).astype(np.int8)
        hy = np.where(w[:, IDX["n_hands"]] > 0, w[:, IDX["hand_cy"]], np.nan)
        hx = np.where(w[:, IDX["n_hands"]] > 0, w[:, IDX["hand_cx"]], np.nan)
        hs = np.hypot(np.diff(hx), np.diff(hy))
        f += [speed.mean(), speed.max(), w[:, IDX["head_dy"]].std(),
              np.abs(np.diff(contact)).sum(), _reversals(hy, 0.12), _reversals(w[:, IDX["wrist_d"]], 0.15),
              np.nanmean(hs) if np.isfinite(hs).any() else 0.0,
              (w[:, IDX["n_hands"]] > 0).mean(), (w[:, IDX["have_face"]] > 0).mean(),
              (dh < 1.0).mean(), (w[:, IDX["d_ear"]] < 1.0).mean()]
        rows.append(f)
        starts.append(a / fps)
    return np.array(rows, np.float32), np.array(starts), names
