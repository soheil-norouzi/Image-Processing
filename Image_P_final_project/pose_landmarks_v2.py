import joblib
import cv2
import mediapipe as mp
import numpy as np

model = joblib.load("/home/soheil/Downloads/fall_detection_model_v2.pkl") # load the trained model

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False) # initialize the pose estimation model
mp_drawing = mp.solutions.drawing_utils # draw the pose landmarksb

webcam = cv2.VideoCapture(0)
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

def extract_features_landmarks(landmarks):
    features = []
    for lm in landmarks.landmark:
        features.extend([lm.x, lm.y, lm.z, lm.visibility])
    return np.array(features).reshape(1,-1)

while True:
    ret, frame = webcam.read() # frame: actual image
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    label = "no pose"

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        features = extract_features_landmarks(results.pose_landmarks)
        prediction = model.predict(features)[0]
        label = "Fall" if prediction == "fall" or prediction == 1 else "Normal"

    cv2.putText(frame, f"Prediction: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0) if label == "Normal" else (0, 0, 255), 2)

    cv2.imshow("Fall Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
