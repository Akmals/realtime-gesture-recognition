# Real-Time Gesture Recognition

A webcam app that recognizes head-and-hand gestures in real time and shows a matching reaction image next to the live video feed.

It uses Google's MediaPipe Pose Landmarker to track body landmarks on each frame, classifies the pose from the geometry of those landmarks, and smooths the result over consecutive frames so the output doesn't flicker.

## Recognized gestures

| Gesture    | How it's detected                                                       |
| ---------- | ----------------------------------------------------------------------- |
| `thinking` | Index finger near the chin                                              |
| `ponder`   | Closed fist near the chin (wrist and index fingertip close together)    |
| `aha`      | Index finger raised above eye level                                     |
| `hehe`     | Smile, measured as mouth width relative to the distance between the eyes |
| `what`     | Neutral face (default when no hand gesture is detected)                 |
| `none`     | No person in frame                                                      |

## How it works

1. **Capture:** OpenCV reads frames from the webcam and mirrors them for a selfie view.
2. **Landmark detection:** each frame goes through MediaPipe's `PoseLandmarker` (lite model), which returns 33 body landmarks with visibility scores.
3. **Classification:** `PoseDetector.determine_pose_state()` checks gestures in priority order using normalized 2D distances between landmarks (fingertips, wrists, nose, eyes, and mouth corners). Hand landmarks are only used when their visibility score is above 0.6, which filters out low-confidence detections.
4. **Temporal smoothing:** the last 5 predictions are kept in a buffer, and the displayed state only changes when all 5 agree. This removes single-frame noise without adding noticeable lag.
5. **Display:** the webcam feed and the matching reaction image are resized to the same height and shown side by side.

## Project structure

```
realtime-gesture-recognition/
├── main.py            # Webcam loop, smoothing, and side-by-side display
├── pose_detector.py   # MediaPipe setup and gesture classification rules
├── meme_manager.py    # Loads reaction images, with placeholders for missing ones
├── test_setup.py      # Checks dependencies and the model without a webcam
├── requirements.txt
└── assets/            # Reaction images: thinking.jpg, aha.jpg, ponder.jpg, what.jpg, hehe.jpg
```

## Setup

Requires Python 3.9+ and a webcam.

```bash
pip install -r requirements.txt
```

Download the MediaPipe pose model into the project folder:

```bash
curl -o pose_landmarker_lite.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
```

Add your reaction images to `assets/` using the names listed above. Any missing image is replaced by a placeholder that shows the detected gesture name.

Verify the setup without a webcam:

```bash
python test_setup.py
```

## Run

```bash
python main.py
```

Press `q` to quit.

## Limitations

- Classification is rule-based: hand-tuned distance thresholds on landmark coordinates, not a trained model. Thresholds were tuned for one person at a typical webcam distance and may need adjusting for other setups.
- Uses 2D coordinates only, so gestures can be misread when the head is turned or the hand is far in front of the face.
- Tracks one person at a time.

## Next steps

- Replace the rule-based classifier with a small trained model: record labeled landmark data for each gesture, train a classifier, and compare its accuracy against the current rules.
- Add more gestures and per-user calibration.
