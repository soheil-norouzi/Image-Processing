# 🧍‍♂️ Real-Time Fall Detection using YOLOv8, MediaPipe, and XGBoost

This repository provides a complete pipeline for **real-time human fall detection**, combining:
- **YOLOv8** for person detection
- **MediaPipe Pose** for extracting body landmarks
- **XGBoost** for classifying posture as `Fall` or `Normal`

---

## 📁 Contents

- `fall_detection.py` – Real-time detection script using webcam
- `pose_detection.ipynb` – Model training notebook (XGBoost)
- `fall_detection_model_v2.pkl` – Trained XGBoost model (you must add this manually)
- `README.md` – Project documentation

---

## 🚀 Quick Start

### 1. Clone the repository

git clone https://github.com/soheil-norouzi/fall-detection.git
cd fall-detection

2. Install requirements
pip install -r requirements.txt

If you don’t have a requirements.txt file yet, you can install manually:
pip install ultralytics mediapipe opencv-python torch numpy joblib xgboost scikit-learn pandas matplotlib seaborn

How It Works
fall_detection.py

    Loads YOLOv8 (yolov8s.pt) to detect people

    Extracts pose landmarks with MediaPipe Pose

    Classifies posture using a trained XGBoost model

    Visualizes bounding boxes and pose landmarks on screen

pose_detection.ipynb

    Loads a .csv of pose landmarks + labels (fall / normal)

    Trains an XGBoost classifier

    Evaluates performance (accuracy, confusion matrix)

    Saves model to fall_detection_model_v2.pkl
