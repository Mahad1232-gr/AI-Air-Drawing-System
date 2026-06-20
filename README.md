# AI Based Air Drawing System
### Using Hand Gesture Recognition and Machine Learning

## 🎯 Project Demo
Draw in the air using hand gestures — no mouse, touchscreen, or stylus needed!

## 📊 Model Performance
- Accuracy: 100%
- Dataset: 18,000 training images (6 gesture classes)
- Model: MobileNetV2 (Transfer Learning)

## ⚙️ Setup
1. Clone repo: `git clone https://github.com/Mahad1232-gr/AI-Air-Drawing-System`
2. Install: `pip install -r requirements.txt`
3. Download `gesture_model.keras` from Kaggle notebook output
4. Place in `model/` folder
5. Run: `python main.py`

## 🤖 Tech Stack
Python | TensorFlow | OpenCV | MediaPipe | Scikit-Learn
# AI Based Air Drawing System Using Hand Gesture Recognition and Machine Learning

A touchless drawing application that lets you draw in the air using hand
gestures captured through a webcam — no mouse, touchscreen, or stylus.

## How It Works

1. **Hand tracking** — MediaPipe detects 21 hand landmarks per frame.
2. **Gesture recognition** — A MobileNetV2 model (transfer learning),
   trained on a 6-class gesture dataset (fist, one, two, three, four, five),
   classifies the hand pose in real time. A geometric finger-count
   cross-check stabilizes low-confidence predictions.
3. **Drawing** — The index fingertip acts as a pen. Movement is smoothed
   with Holt's Double Exponential Smoothing to remove hand-tremor jitter.
4. **Shape recognition (AI Snap)** — A KNN + Random Forest ensemble,
   trained on synthetically generated circles/squares/triangles/lines,
   classifies your rough stroke and replaces it with a clean shape.
5. **Lock system** — Holding a gesture for ~1 second locks that mode so
   stray hand movements can't accidentally trigger other actions. This
   was the main reliability fix needed after early testing showed
   frequent accidental color changes, canvas clears, and missed saves.

## Gestures

| Gesture | Action |
|---|---|
| Fist (held) | Clear Canvas |
| 1 Finger | Draw |
| 2 Fingers | AI Snap |
| 3 Fingers | Change Color |
| 4 Fingers (held) | Save Drawing |
| 5 Fingers (held, while locked) | Unlock |

## Keyboard (lock control + brush size only — never replaces a gesture)

| Key | Action |
|---|---|
| L | Lock current mode |
| U | Unlock |
| B | Increase brush size |
| N | Decrease brush size |
| Q | Quit |

## Project Structure

```
AI_Air_Drawing/
├── main.py                 # App entry point, lock system, UI loop
├── gesture_detector.py     # MediaPipe + CNN gesture classification
├── shape_classifier.py     # KNN + Random Forest shape recognition
├── ai_snap.py               # Converts rough strokes into clean shapes
├── model/
│   └── gesture_model.h5    # Trained CNN (place file here after training)
├── saved_drawings/         # Output PNGs land here
├── utils/
│   ├── drawing_utils.py    # Canvas, UI rendering, status bar, guide
│   └── smoothing.py        # Holt's double exponential smoothing
├── colab/
│   ├── 01_train_gesture_model.py
│   ├── 02_evaluate_model.py
│   └── 03_export_model.py
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

Place your trained `gesture_model.h5` inside `model/`, then run:

```bash
python main.py
```

## Training Your Own Gesture Model

1. Organize a dataset as:
   ```
   dataset/
   ├── train/{fist,one,two,three,four,five}/
   └── valid/{fist,one,two,three,four,five}/
   ```
2. Open `colab/01_train_gesture_model.py` in Google Colab, set
   `DATASET_DIR`, and run all cells. This trains MobileNetV2 via
   transfer learning + fine-tuning and saves `gesture_model.h5`.
3. Run `colab/02_evaluate_model.py` for accuracy/precision/recall/F1
   and a confusion matrix.
4. Run `colab/03_export_model.py` to also produce a smaller, faster
   `gesture_model.tflite`.
5. Copy `gesture_model.h5` into `model/` in this project.

## Why KNN + Random Forest for shapes, but a CNN for gestures?

Shape recognition (circle/square/triangle/line) is a small, 4-class
problem that hand-crafted geometric features (compactness, convexity,
aspect ratio, etc.) already describe very accurately — a CNN would add
latency and require image data for no real accuracy gain, and AI Snap
needs to feel instant. Gesture recognition from raw hand images benefits
more from a CNN's ability to generalize across lighting, angle, and skin
tone variation, which simple finger-counting struggles with.
