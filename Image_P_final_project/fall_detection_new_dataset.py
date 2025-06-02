from ultralytics import YOLO
import numpy as np
import mediapipe as mp
import joblib
import cv2
import torch
import os # Import os for better path handling if needed

# --- Model Loading ---
yolo = YOLO('yolov8n.pt')
if torch.cuda.is_available():
    yolo.to('cuda')
    print("YOLO model moved to GPU.")
else:
    print("CUDA not available. YOLO model will run on CPU.")

print(f"YOLO model device: {yolo.device}")

# Path to your trained fall detection model (the .pkl file)
# Ensure this path is correct
model_path = "/home/soheil/Downloads/fall_detection_model.pkl" # Adjust if your .pkl is elsewhere

try:
    xgb_model = joblib.load(model_path)
    print(f"Fall detection model loaded successfully from: {model_path}")
except FileNotFoundError:
    print(f"Error: The model file was not found at {model_path}.")
    print("Please ensure the model file is in the correct directory.")
    exit()
except Exception as e:
    print(f"Error loading the model: {e}")
    exit()

# --- MediaPipe Pose Initialization ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)
mp_drawing = mp.solutions.drawing_utils

# --- Label Mapping (Crucial for interpreting model output) ---
# This mapping MUST match the one used during your model training (0: 'Fall detected', 1: 'Walking', 2: 'Sitting')
prediction_to_label = {
    0: 'Fall detected',
    1: 'Walking',
    2: 'Sitting'
}

# --- Function to extract pose landmarks ---
def extract_pose_landmarks(image):
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_img)
    if not results.pose_landmarks:
        return None, None

    landmarks = results.pose_landmarks.landmark
    features = []
    for lm in landmarks:
        features.extend([lm.x, lm.y, lm.z, lm.visibility])

    # Ensure the feature vector has the expected length (33 landmarks * 4 values/landmark = 132)
    if len(features) != 132:
        print(f"Warning: Expected 132 features, but got {len(features)}. Skipping this person for pose prediction.")
        return None, None

    return np.array(features).reshape(1,-1), results.pose_landmarks

# --- Webcam Initialization ---
webcam = cv2.VideoCapture(0)
if not webcam.isOpened():
    print("Error: Could not open webcam.")
    exit()

webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
print(f"Webcam opened. Resolution set to {webcam.get(cv2.CAP_PROP_FRAME_WIDTH)}x{webcam.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

# --- Main Loop for Real-time Detection ---
while webcam.isOpened():
    ret, frame = webcam.read()
    if not ret:
        print("Failed to grab frame, exiting...")
        break
    
    frame = cv2.flip(frame, 1) # Mirror horizontally for natural webcam use

    # YOLO object detection for 'person' class (class ID 0)
    # Using conf=0.4 to ensure reasonable confidence in detections
    results = yolo.predict(frame, classes=[0], conf=0.4, verbose=False)
    detections = results[0].boxes.xyxy.cpu().numpy() # Extract bounding box coordinates

    for det in detections:
        x1, y1, x2, y2 = map(int, det) # Convert coordinates to integers
        
        # Ensure bounding box coordinates are within frame boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        
        person_img = frame[y1:y2, x1:x2] # Extract region of interest (person's image)
        
        # Skip very small or empty person images to avoid errors
        if person_img.shape[0] < 50 or person_img.shape[1] < 50:
            continue

        # Extract pose landmarks from the person's ROI
        features, pose_landmarks_to_draw = extract_pose_landmarks(person_img)
        
        # Default labels and colors
        label = "No Pose"
        color = (255, 0, 0) # Blue for 'No Pose'

        # If pose landmarks were successfully extracted, predict fall status
        if features is not None:
            try:
                # The model predicts a numerical ID (0, 1, or 2)
                numerical_prediction = xgb_model.predict(features)[0]
                
                # Map the numerical prediction to the human-readable label
                predicted_activity = prediction_to_label.get(numerical_prediction, "Unknown")
                
                # Determine the label and color based on the predicted activity
                if predicted_activity == 'Fall detected':
                    label = "FALL DETECTED!"
                    color = (0, 0, 255) # Red for Fall
                elif predicted_activity == 'Walking':
                    label = "Walking"
                    color = (0, 255, 0) # Green for Walking
                elif predicted_activity == 'Sitting':
                    label = "Sitting"
                    color = (255, 255, 0) # Cyan/Yellow for Sitting
                else: # Should ideally not happen if mapping is correct
                    label = "Unknown Activity"
                    color = (0, 165, 255) # Orange for Unknown

                # Draw MediaPipe pose landmarks on the original frame
                # Create a black image with the same size as the person's ROI
                roi_display = np.zeros_like(person_img)
                mp_drawing.draw_landmarks(
                    roi_display,
                    pose_landmarks_to_draw,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2), # Red dots
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2), # White lines
                )
                # Blend the pose landmarks onto the original frame's ROI
                frame[y1:y2, x1:x2] = cv2.addWeighted(frame[y1:y2, x1:x2], 1, roi_display, 0.7, 0)

            except Exception as e:
                label = "Prediction Error"
                color = (0, 165, 255) # Orange
                print(f'Prediction error for person: {e}')

        # Draw a bounding box around the person and display the label
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # Display the final frame
    cv2.imshow("Multi-Person Activity Detection", frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
webcam.release()
cv2.destroyAllWindows()
print("Webcam released and windows closed.")