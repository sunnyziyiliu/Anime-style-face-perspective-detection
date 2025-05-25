# Anime-Style Face Perspective Detection

An interactive Streamlit app that detects 24 facial keypoints on anime-style faces, allows manual offset corrections, evaluates facial symmetry & perspective, and visualizes results.

---
## Source model

(https://github.com/kanosawa/anime_face_landmark_detection)

---

## Live Demo

[View the app online](https://sunnyziyiliu-anime-style-face-perspective-detection-app-rajcps.streamlit.app/)

---

## Features

- **Automatic Keypoint Detection**  
  Uses a pretrained Cascaded Face Alignment (CFA) model to predict 24 landmarks on anime faces.

- **Manual Offset Corrections**  
  Sidebar UI lets you add/remove multiple `(index, Δx, Δy)` adjustments, then apply them all at once.

- **Alignment Checks**  
  - Eye parallelism  
  - Nose–lips–chin colinearity  
  - (Optional) Eyebrow parallelism  
  - (Optional) Overall feature alignment  

- **Perspective Ratios & Face Orientation**  
  Computes projection-based ratios for eye head/middle/tail and mouth, identifies if the face is turned left/right or frontal, and highlights any perspective errors.

- **Color-Coded Results**  
  - Green text for correct  
  - Red text for errors  
  - Gray text for skipped checks  
  - Black text for neutral orientation messages

---

## Requirements

Captured via `requirements.txt`:

```text
streamlit>=1.10.0
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.7.0
numpy>=1.23.0
Pillow>=9.0.0
altair>=4.0
