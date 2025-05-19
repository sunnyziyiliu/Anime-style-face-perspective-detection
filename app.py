import os
import cv2
import torch
import numpy as np
import math
from urllib.request import urlretrieve
import streamlit as st
from torchvision import transforms
from PIL import Image
from CFA import CFA
from eval_utils import evaluate, fit_midline, perp_dist
from eval_utils import suggest_adjustments, adjust_landmark_pair
# ——— HEAD —————————————————————
st.set_page_config(
    page_title="Anime Face Drawing Feedback",
    layout="wide"
)
st.title("Anime Face Drawing Feedback")
st.write("Using 24 keypoints, detect the facial feature and verify if the perspective and proportions are accurate.")
st.markdown(
    "AI Model Source："
    "[Anime Face Landmark Detection by kanosawa]"
    "(https://github.com/kanosawa/anime_face_landmark_detection)"
)

# ——— OFFSET —————————————————————
if "offsets" not in st.session_state:
    st.session_state.offsets = []      # [{ "idx":int, "dx":int, "dy":int }, ...]
    st.session_state.apply_all = False

# ——— SIDEBAR —————————————————————
st.sidebar.header("Manual Keypoints Offsets")


st.sidebar.write("If the AI’s detected keypoints are inaccurate, please manually adjust them here.")


col_add, col_rm = st.sidebar.columns(2)
if col_add.button("Add offset"):
    st.session_state.offsets.append({"idx": 0, "dx": 0, "dy": 0})
if col_rm.button("Remove offset") and st.session_state.offsets:
    st.session_state.offsets.pop()

for i, off in enumerate(st.session_state.offsets):
    c1, c2, c3 = st.sidebar.columns(3)
    with c1:
        st.session_state.offsets[i]["idx"] = st.number_input(
            f" Landmark", min_value=0, max_value=23,
            key=f"idx_{i}", value=off["idx"]
        )
    with c2:
        st.session_state.offsets[i]["dx"] = st.number_input(
            f" X(right+/left-)", key=f"dx_{i}", value=off["dx"]
        )
    with c3:
        st.session_state.offsets[i]["dy"] = st.number_input(
            f"Y (down+/up-)", key=f"dy_{i}", value=off["dy"]
        )


if st.sidebar.button("Apply all offsets"):
    st.session_state.apply_all = True
else:
    st.session_state.apply_all = False

skip_expr = st.sidebar.checkbox("Exaggerated Expression (skip brow checks)")



# ——— LOADING MODEL —————————————————————
CASCADE_FILE = "lbpcascade_animeface.xml"
if not os.path.exists(CASCADE_FILE):
    urlretrieve(
        "https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface/master/lbpcascade_animeface.xml",
        CASCADE_FILE
    )
detector = cv2.CascadeClassifier(CASCADE_FILE)

# ——— LOADING KEYPOINTS MODEL —————————————————————
@st.cache_resource
def load_model():
    ckpt = "checkpoint_landmark_191116.pth.tar"
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CFA(output_channel_num=25)
    state = torch.load(ckpt, map_location=device)
    sd = state.get("state_dict", state)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return model, device

model, device = load_model()

# ——— RESCALE IMAGE —————————————————————
IMG_W = 128
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ——— MAIN —————————————————————
uploaded = st.file_uploader("Upload an anime face image:", type=["png","jpg","jpeg"])
if not uploaded:
    st.stop()

# 1) DECODE TO OPENCV
buf     = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
img_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
h0, w0  = img_bgr.shape[:2]

# # 2) SHOW IMAGE
# st.image(
#     cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
#     caption="Original Image",
#     use_container_width=False,
#     width=w0
# )

# 3) FACE DETECTION
gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(24,24))
if len(faces) == 0:
    st.warning("No face detected.")
    st.stop()

x_, y_, w_, h_ = faces[0]
x0 = max(int(x_ - w_/8), 0)
rx = min(int(x_ + 9*w_/8), w0)
y0 = max(int(y_ - h_/4), 0)
by = min(int(y_ + h_), h0)
w = rx - x0
h = by - y0

# 4) Crop & Resize
crop    = img_bgr[y0:by, x0:rx]
face_bgr = cv2.resize(crop, (IMG_W, IMG_W), interpolation=cv2.INTER_CUBIC)
face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
inp      = transform(Image.fromarray(face_rgb)).unsqueeze(0).to(device)

# 5) heatmaps
heatmaps = model(inp)[-1].cpu().detach().numpy()[0]

# 6) Map-back --> landmarks
landmarks = []
for i in range(24):
    hm      = cv2.resize(heatmaps[i], (IMG_W, IMG_W), interpolation=cv2.INTER_CUBIC)
    idx     = int(hm.argmax())
    y_idx, x_idx = divmod(idx, IMG_W)
    X = int(x0 + x_idx * w / IMG_W)
    Y = int(y0 + y_idx * h / IMG_W)
    landmarks.append([X, Y])

# 7) APPLE OFFSET
if st.session_state.apply_all:
    for off in st.session_state.offsets:
        i = off["idx"]
        landmarks[i][0] += off["dx"]
        landmarks[i][1] += off["dy"]

# 8) MARK ON BGR buffer
annot = img_bgr.copy()
for i, (X, Y) in enumerate(landmarks):
    cv2.circle(annot, (X, Y), 2, (0,0,255), -1)
    cv2.putText(annot, str(i), (X+3, Y-3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)

# 9) SHOW THE MARKED IMAGE
st.image(
    cv2.cvtColor(annot, cv2.COLOR_BGR2RGB),
    caption="Annotated Landmarks (with offsets applied)",
    use_container_width=False,
    width=w0
)
# USE EVAL
results, ratios = evaluate(landmarks, skip_expr=skip_expr)

# SHOW EVAL
#
# for line in results:
#     st.markdown(f"- {line}")



st.subheader("Alignment Checks")
for text, status in results[:-2]: 
    if status == "correct":
        color = "green"
    elif status == "error":
        color = "red"
    else:
        color = "gray"
    st.markdown(f"<span style='color:{color}'>{text}</span>", unsafe_allow_html=True)


st.subheader("Perspective Ratios")
# for name, val in ratios.items():
#     st.write(f"- {name}: {val:.3f}")

overall_text, _ = results[-2]
st.markdown(f"<span style='color:black'>{overall_text}</span>", unsafe_allow_html=True)

persp_text, persp_status = results[-1]
persp_color = "green" if persp_status == "correct" else "red"
st.markdown(f"<span style='color:{persp_color}'>{persp_text}</span>", unsafe_allow_html=True)




# adjust advice image version
mid_idxs = [9,21,23,1]
pts_mid = np.array([landmarks[i] for i in mid_idxs], dtype=np.float32)
mean, dir_vec = fit_midline(pts_mid)
med = float(np.median([ratios[n] for n in ratios if not np.isnan(ratios[n])]))

# all adjust option
suggests = suggest_adjustments(results, ratios, landmarks)

if suggests:
    st.subheader("Visual modification suggestions")
    # option box
    sel = st.selectbox("Select an error to adjust", list(suggests.keys()))
    mode = st.radio("Adjustment mode", ["Lock Left", "Lock Right", "Average"])

    # find the pair
    item = suggests[sel]
    adjusted = []
    if item["type"] == "persp":
        for (i,j,name) in item["pairs"]:
            adjusted += adjust_landmark_pair(
                landmarks, mean, dir_vec, i, j, mode, med
            )
    elif item["type"] in ("eye","brow"):
        for (i,j,name) in item["pairs"]:
            adjusted += adjust_landmark_pair(
                landmarks, mean, dir_vec, i, j, mode, med
            )

    # draw
    vis = annot.copy()
    for idx, nx, ny in adjusted:
        cv2.circle(vis, (nx, ny), 5, (0,0,255), -1)
        cv2.putText(vis, f"{idx}", (nx+5, ny-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption="Adjustment Suggestions")
else:
    st.markdown(
    "<span style='color: blue; font-size: 2rem;'>Great! Your face drawing is very accurate!</span>",
    unsafe_allow_html=True
)
