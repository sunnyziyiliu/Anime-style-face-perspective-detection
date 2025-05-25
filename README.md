# Anime-Style Face Perspective Detection

This project uses AI to detect 24 key points on an anime face, applies geometric methods to evaluate symmetry axes, parallel lines, and perspective ratios, and then gives creators both text and visual feedback so they can quickly spot and fix structural errors in their work.


---
## Source model

(https://github.com/kanosawa/anime_face_landmark_detection)

---

## Live Demo

[View the app online](https://sunnyziyiliu-anime-style-face-perspective-detection-app-rajcps.streamlit.app/)

---

## Overall Workflow

When a user uploads an anime face image, the key-point detection AI locates the 24 predefined facial landmarks and returns their coordinates. The system then draws each point and its index onto the image and displays it on a webpage. On the left, a “Key Point Offset” panel lets users fine-tune any misplaced points by entering the index and offset values. Once all coordinates are collected, the system calculates facial alignment and perspective. By checking whether lines between eye corners and brow points are parallel and whether landmarks lie on a shared midline, it assesses alignment. It also compares each landmark’s distance to the midline—if any left-right pair violates the expected perspective ratio, the system flags it. Finally, the page outputs a text report (green for correct items, red for errors, black for neutral descriptions) and, for any errors, generates three correction options—“Lock Left,” “Lock Right,” or “Average”—allowing users to select the point to fix or average; the system then computes the correct coordinate and redraws it on the image.
