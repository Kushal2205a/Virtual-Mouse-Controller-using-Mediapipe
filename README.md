# MediaPipe Gesture + Gaze Mouse Controller

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-brightgreen)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands%20%2B%20FaceMesh-orange)

Control your mouse using hand gestures (MediaPipe Hands) with an optional gaze assist (MediaPipe FaceMesh iris landmarks). Includes click/drag and a few media/system shortcuts.

<!--
## Demo
Add your media files under `assets/` and uncomment:

![Demo](assets/demo.gif)
![Screenshot](assets/screenshot.png)
-->

---

## Features

- **Cursor movement** using index fingertip mapping + smoothing
- **Clicks & drag** via gesture states (debounced to reduce accidental clicks)
- **Optional gaze assist** (toggle at runtime) blended into cursor position
- **Data collection mode** to record samples for training gesture models

---

## Gesture Map (default)

> Gesture labels come from `model/keypoint_classifier/keypoint_classifier_label.csv`. If you retrain/rename labels, update the mappings in `app.py`.

| Gesture | Action |
|--------|--------|
| `Open` | Move cursor |
| `Pointer` | Left click-and-hold (drag) |
| `Close` | Left click |
| `OK` | Right click |
| `Peace` | Double click |
| `Thumbsup` | Play/Pause (fallback: Space) |
| `Thumbsdown` | Close window (Alt+F4) |
| `Call` | App switch (Alt+Tab) |
| `threefingersup` | Volume up |
| `rock` | Volume down |

> Note: `Open` vs `OK` is stabilized using a thumb–index distance ratio inside the script.

---

## Keyboard Controls (while the window is focused)

- `ESC` — quit
- `g` — toggle gaze assist
- `c` — calibrate gaze center to current eye position
- `y` — invert gaze Y
- `[` / `]` — decrease / increase gaze gain
- Dataset modes:
  - `n` — normal mode
  - `k` — log keypoint samples
  - `h` — log point-history samples
  - `0`–`9` — class label to write for the current mode

---

## Requirements

- Python 3.8+
- A webcam
- OS permissions for controlling the mouse (macOS may prompt for Accessibility permission)

Install dependencies:

```bash
pip install opencv-python mediapipe numpy pyautogui screeninfo tensorflow
```

> If you plan to train the model using `keypoint_classifcation.py`, you’ll also want:
> `pandas scikit-learn matplotlib seaborn`

---

## Run

```bash
python app.py
```

Optional arguments:

```bash
python app.py --device 0 --width 960 --height 540
```

---
## Data Collection & Training (optional)

The app can write training rows directly while it’s running:

1. Start the app
2. Press:
   - `k` to log keypoints into `model/keypoint_classifier/keypoint.csv`
   - `h` to log point history into `model/point_history_classifier/point_history.csv`
3. Press `0`–`9` to save samples for that class.
4. Train the classifier:

```bash
python keypoint_classifcation.py
```

This produces a `keypoint_classifier.tflite` used at runtime.

---
