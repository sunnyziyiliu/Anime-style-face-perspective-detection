import numpy as np
import math

def slope(p1, p2):
    x1,y1 = p1; x2,y2 = p2
    return np.inf if x2==x1 else (y2-y1)/(x2-x1)

def foot_of_perp(P, A, B):
    A = np.array(A, dtype=np.float32)
    B = np.array(B, dtype=np.float32)
    P = np.array(P, dtype=np.float32)
    AB = B - A
    t  = np.dot(P - A, AB) / np.dot(AB, AB)
    return A + t * AB

def evaluate(landmarks, skip_expr=False,
             tol_deg=5.0, tol_col=0.1):
    """  
    输入：
      landmarks: List[(x,y)] 24 个点  
      skip_expr: 是否跳过眉毛&整体检查  
    返回：
      results: List[str]   — 文本评估结果  
      ratios:  Dict[name->float] — 透视比例  
    """
    results = []

    # 1: eye parallel
    eye_segs = {
        "lower eyelid": (10,17),
        "upper eyelid": (11,16),
        "pupil line":   (14,19),
        "eyeball axis": (23,18),
    }
    angles = {}
    for name,(i,j) in eye_segs.items():
        dx = landmarks[j][0] - landmarks[i][0]
        dy = landmarks[j][1] - landmarks[i][1]
        angles[name] = math.degrees(math.atan2(dy, dx))
    ref_eye = float(np.median(list(angles.values())))
    mis_eyes = [n for n,a in angles.items()
                if abs((a - ref_eye + 180) % 360 - 180) > tol_deg]
    results.append(f"Eye slant: {'; '.join(mis_eyes) or 'Correct'}")

    # 2: middle line
    idxs = [9,21,23,1]
    p1,p2,p3,p4 = [landmarks[i] for i in idxs]
    L = np.hypot(p2[0]-p1[0], p2[1]-p1[1])
    ok_col = False
    if L>0:
        dist = lambda pt: abs((p2[0]-p1[0])*(pt[1]-p1[1])
                              - (p2[1]-p1[1])*(pt[0]-p1[0]))/L
        ok_col = (dist(p3)/L <= tol_col and dist(p4)/L <= tol_col)
    results.append(f"Midline: {'Correct' if ok_col else 'Incorrect'}")

    # 3: eyebrown
    if skip_expr:
        results.append("Eyebrow check skipped")
        ref_brow, mis_brow = 0, []
    else:
        brow_segs = {
            "brow tail":   (2,6),
            "brow middle": (4,7),
            "brow head":   (5,6),
        }
        slopes_brow = {}
        for name,(i,j) in brow_segs.items():
            slopes_brow[name] = slope(landmarks[i], landmarks[j])
        fb = [s for s in slopes_brow.values() if np.isfinite(s)]
        ref_brow = float(np.median(fb)) if fb else 0
        mis_brow = [n for n,s in slopes_brow.items()
                    if (np.isfinite(s) and abs(s-ref_brow)/abs(ref_brow or 1) > tol_col)
                       or (not np.isfinite(s) and np.isfinite(ref_brow))]
        results.append(f"Eyebrows: {'; '.join(mis_brow) or 'Correct'}")

    # 4 overall
    if skip_expr:
        results.append("Overall check skipped")
    else:
        if not mis_eyes and ok_col and not mis_brow:
            mouth_s = slope(landmarks[20], landmarks[22])
            ref_all = np.median([ref_eye, ref_brow, mouth_s])
            mis_all = []
            for name,s in zip(["eye","brow","mouth"], [ref_eye, ref_brow, mouth_s]):
                if abs(s-ref_all)/abs(ref_all or 1) > tol_col:
                    mis_all.append(name)
            results.append(f"Overall: {'; '.join(mis_all) or 'Correct'}")
        else:
            results.append("Overall check skipped due to earlier misalignment")

    # ——— perspective —————————————————————
    pairs = [
        (12,15,"Eye–Head"),
        (11,16,"Eye–Middle"),
        (10,17,"Eye–Tail"),
        (20,22,"Mouth")
    ]
    P1 = landmarks[1]
    ratios = {}
    for i,j,name in pairs:
        F = foot_of_perp(P1, landmarks[i], landmarks[j])
        dA = np.linalg.norm(F - landmarks[i])
        dB = np.linalg.norm(F - landmarks[j])
        ratios[name] = float(dA/dB) if dB > 1e-6 else float("nan")

    return results, ratios
