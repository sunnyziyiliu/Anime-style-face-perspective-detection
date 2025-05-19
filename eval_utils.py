# eval_utils.py
import numpy as np
import math

def slope(p1, p2):
    """
    find slope: np.inf
    p1, p2: (x, y)
    """
    x1, y1 = p1
    x2, y2 = p2
    return np.inf if x2 == x1 else (y2 - y1) / (x2 - x1)

def fit_midline(points):
    """
    PCA --> middle line
    points: np.array of shape (N,2)
    return (mean, dir_vec)，dir_vec as unit direction。
    """
    mean = points.mean(axis=0)
    X = points - mean
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    dir_vec = Vt[0] / np.linalg.norm(Vt[0])
    return mean, dir_vec

def perp_dist(pt, mean, dir_vec):
    """
    calculate pt to line (mean + t*dir_vec) perpendicular
    """
    return abs(np.cross(dir_vec, pt - mean))

def evaluate(landmarks, skip_expr=False,
             tol_deg=5.0, tol_col=0.1):
    """
    

    input：
      landmarks: List[(x, y)] 
      skip_expr: if skip
      tol_deg:   tolorence
      tol_col:   tolorence

    return：
      results: List[(text: str, status: str)]
      ratios:  Dict[name: str → float]
    """
    results = []

    # ——— 1. eye alignment —————————————————————
    eye_segs = {
        "eyelid head":(12,15),
        "eyelid tail": (10, 17),
        "eyelid middle": (11, 16),
        "lower eyelid": (13, 18),
        "pupile":   (14, 19),
    }
    angles = {}
    for name, (i, j) in eye_segs.items():
        dx = landmarks[j][0] - landmarks[i][0]
        dy = landmarks[j][1] - landmarks[i][1]
        angles[name] = math.degrees(math.atan2(dy, dx))
    ref_eye = float(np.median(list(angles.values())))

    mis_eyes = []
    for name, (i, j) in eye_segs.items():
        a = angles[name]
        if abs((a - ref_eye + 180) % 360 - 180) > tol_deg:
            yi, yj = landmarks[i][1], landmarks[j][1]
            if yi < yj:
                direction = "left too high, right too low"
            else:
                direction = "left too low, right too high"
            mis_eyes.append(f"{name} ({direction})")

    if mis_eyes:
        results.append((f"Eye slant: {', '.join(mis_eyes)}", "error"))
    else:
        results.append(("Eye: Correct", "correct"))

    # ——— 2. midline（PCA）—————————————————
    mid_idxs = [9, 21, 23, 1]  # 鼻尖、上嘴唇中、下嘴唇中、下巴
    pts_mid = np.array([landmarks[i] for i in mid_idxs], dtype=np.float32)
    mean, e = fit_midline(pts_mid)
    L = np.linalg.norm(pts_mid[0] - pts_mid[-1])
    dists = [perp_dist(p, mean, e) for p in pts_mid]
    if L > 0 and max(dists) / L <= tol_col:
        results.append(("Midline: Correct", "correct"))
    else:
        results.append(("Midline: Incorrect", "error"))

    # ——— 3. eyebrow align —————————————————————
    if skip_expr:
        results.append(("Eyebrow check skipped", "skipped"))
        mis_brow = []
    else:
        brow_segs = {
            "brow tail":   (3, 8),
            "brow middle": (4, 7),
            "brow head":   (5, 6),
        }
        slopes_brow = {
            name: slope(landmarks[i], landmarks[j])
            for name, (i, j) in brow_segs.items()
        }
        fb = [s for s in slopes_brow.values() if np.isfinite(s)]
        ref_brow = float(np.median(fb)) if fb else 0

        mis_brow = []
        for name, (i, j) in brow_segs.items():
            s = slopes_brow[name]
            cond = (np.isfinite(s) and abs(s - ref_brow) / abs(ref_brow or 1) > tol_col) \
                   or (not np.isfinite(s) and np.isfinite(ref_brow))
            if cond:
                yi, yj = landmarks[i][1], landmarks[j][1]
                if yi < yj:
                    direction = "left too high, right too low"
                else:
                    direction = "left too low, right too high"
                mis_brow.append(f"{name} ({direction})")

        if mis_brow:
            results.append((f"Eyebrows slant: {', '.join(mis_brow)}", "error"))
        else:
            results.append(("Eyebrows: Correct", "correct"))

    # # ——— 4. all —————————————————————
    # if skip_expr:
    #     results.append(("Overall check skipped", "skipped"))
    # else:
    #     if not mis_eyes and not mis_brow and (L > 0 and max(dists)/L <= tol_col):
    #         mouth_s = slope(landmarks[20], landmarks[22])
    #         ref_all = np.median([ref_eye, ref_brow, mouth_s])
    #         mis_all = [
    #             name for name, s in zip(
    #                 ["eye", "brow", "mouth"],
    #                 [ref_eye, ref_brow, mouth_s]
    #             )
    #             if abs(s - ref_all) / abs(ref_all or 1) > tol_col
    #         ]
    #         if mis_all:
    #             results.append((f"Overall misalignment: {', '.join(mis_all)}", "error"))
    #         else:
    #             results.append(("Overall Alignment: Correct", "correct"))
    #     else:
    #         results.append(("Overall check skipped due to earlier misalignment", "skipped"))

    # ——— 5. perspective ——————————————————
    pairs = [
        (12, 15, "Eye–Head"),
        (11, 16, "Eye–Middle"),
        (10, 17, "Eye–Tail"),
        (20, 22, "Mouth"),
    ]
    ratios = {}
    for i, j, name in pairs:
        di = perp_dist(np.array(landmarks[i]), mean, e)
        dj = perp_dist(np.array(landmarks[j]), mean, e)
        ratios[name] = float(di / dj) if dj > 1e-6 else float("nan")

    tol_persp = tol_col
    valid = [r for r in ratios.values() if not np.isnan(r)]
    med = float(np.median(valid)) if valid else 1.0

    # 5.1 face
    if med > 1 + tol_persp:
        results.append(("Face turned right", "neutral"))
    elif med < 1 - tol_persp:
        results.append(("Face turned left", "neutral"))
    else:
        results.append(("Frontal face", "neutral"))

    # 5.2 wrong perspective
    mis = []
    for name, r in ratios.items():
        if np.isnan(r):
            continue
        if abs(r - med) / med > tol_persp:
            if r > med:
                text = f"{name}: left too far from midline, right too close to midline"
            else:
                text = f"{name}: left too close to midline, right too far from midline"
            mis.append(text)

    if mis:
        results.append(("Perspective issues: " + "; ".join(mis), "error"))
    else:
        results.append(("Perspective correct", "correct"))

    return results, ratios


# visualization adjustment

def suggest_adjustments(results, ratios, landmarks, tol_col=0.1):
    """
     evaluate()  results/ratios find wrong pair：
      { 
        err_text: {
          "type": "persp"|"eye"|"brow"|"midline",
          "pairs": [ (i,j,name), ... ],     # influneced point pair
          "detail": [desc strings],         # description
        }, ...
      }
    """
    suggestions = {}

    for text, status in results:
        if status != "error":
            continue
        if text.startswith("Perspective"):
            # e.g. "Perspective issues: Eye–Head: left too close…, right too far…; Mouth: …"
            raw = text.replace("Perspective issues: ", "")
            for part in raw.split("; "):
                name, desc = part.split(": ", 1)
                # 找对称点对索引
                pair_map = {
                    "Eye–Head":   (12, 15, "Eye–Head"),
                    "Eye–Middle": (11, 16, "Eye–Middle"),
                    "Eye–Tail":   (10, 17, "Eye–Tail"),
                    "Mouth":      (20, 22, "Mouth"),
                }
                if name not in pair_map:
                    continue
                i, j, _ = pair_map[name]
                key = f"persp::{name}"
                suggestions[key] = {
                    "type": "persp",
                    "pairs": [(i,j,name)],
                    "detail": [desc],
                }
        elif text.startswith("Eye slant"):
            # e.g. "Eye slant: lower eyelid (left too high...), pupil line (...)"
            raw = text.replace("Eye slant: ", "")
            for seg in raw.split(", "):
                nm, _, desc = seg.partition(" (")
                desc = desc.rstrip(")")

                eye_segs = {
                    "lower eyelid": (10,17),
                    "upper eyelid": (11,16),
                    "pupil line":   (14,19),
                    "eyeball axis": (13,18),
                }
                if nm not in eye_segs:
                    continue
                i,j = eye_segs[nm]
                key = f"eye::{nm}"
                suggestions[key] = {
                    "type": "eye",
                    "pairs": [(i,j,nm)],
                    "detail": [desc],
                }
        elif text.startswith("Eyebrows slant"):
            raw = text.replace("Eyebrows slant: ", "")
            for seg in raw.split(", "):
                nm, _, desc = seg.partition(" (")
                desc = desc.rstrip(")")
                brow_segs = {
                    "brow tail":   (3,8),
                    "brow middle": (4,7),
                    "brow head":   (5,6),
                }
                if nm not in brow_segs:
                    continue
                i,j = brow_segs[nm]
                key = f"brow::{nm}"
                suggestions[key] = {
                    "type": "brow",
                    "pairs": [(i,j,nm)],
                    "detail": [desc],
                }
        # elif text.startswith("Midline"):
        #     suggestions["midline"] = {
        #         "type": "midline",
        #         "pairs": [],  
        #         "detail": [text],
            # }

    return suggestions


def adjust_landmark_pair(landmarks, mean, dir_vec, i, j, mode, med):


    pi = np.array(landmarks[i])
    pj = np.array(landmarks[j])
    di = perp_dist(pi, mean, dir_vec)
    dj = perp_dist(pj, mean, dir_vec)


    if mode == "Lock Left":
        ti = di
        tj = di / med
    elif mode == "Lock Right":
        tj = dj
        ti = dj * med
    else:  # Average
        avg = (di + dj) / 2
        ti = tj = avg

    def project(idx, target):
        P = np.array(landmarks[idx])
        t = np.dot(P - mean, dir_vec)
        F = mean + t * dir_vec
        side = np.sign(np.cross(dir_vec, P - mean))

        perp = np.array([-dir_vec[1], dir_vec[0]])
        newP = F + side * target * perp
        return idx, int(newP[0]), int(newP[1])

    return [project(i, ti), project(j, tj)]
