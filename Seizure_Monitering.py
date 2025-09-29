import cv2
import mediapipe as mp
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import traceback

# === EMAIL CONFIGURATION ===
SENDER_EMAIL = "clinicruby86@gmail.com"
APP_PASSWORD = "tjjg kjvh hfbk" # your 8 digit key 
RECEIVER_EMAIL = "clinicruby86@gmail.com"

def send_email_alert(subject, body):
    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECEIVER_EMAIL
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(SENDER_EMAIL, APP_PASSWORD)
            server.send_message(msg)
        print("✅ Email alert sent successfully!")
        return True
    except Exception as e:
        print("❌ Failed to send email:")
        traceback.print_exc()
        return False

# Mediapipe Setup
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose = mp_pose.Pose()
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Video capture
cap = cv2.VideoCapture(0)

# Motion parameters
prev_body_landmarks = None
prev_hand_positions = None
jerk_threshold = 15
hand_jerk_threshold = 20
frequency_threshold = 10
jerk_counter = 0
seizure_alert = False
alert_sent = False

# Major body joints
body_joints = [mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_SHOULDER,
               mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP,
               mp_pose.PoseLandmark.RIGHT_HIP]

# Extract hand positions (wrist and index tip)
def extract_hand_positions(results):
    positions = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            positions.append(np.array([wrist.x, wrist.y]))
            positions.append(np.array([index_tip.x, index_tip.y]))
    return np.array(positions) if positions else None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(rgb_frame)
    hands_results = hands.process(rgb_frame)

    # Draw landmarks
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    body_movement = 0
    hand_movement = 0

    # Analyze body movement
    if pose_results.pose_landmarks:
        landmarks = np.array([[lm.x, lm.y] for lm in pose_results.pose_landmarks.landmark])
        key_body_landmarks = landmarks[[joint.value for joint in body_joints]]

        if prev_body_landmarks is not None:
            delta = np.linalg.norm(key_body_landmarks - prev_body_landmarks, axis=1)
            body_movement = np.mean(delta) * 1000  # Scale factor

        prev_body_landmarks = key_body_landmarks

    # Analyze hand movement
    current_hand_positions = extract_hand_positions(hands_results)
    if (current_hand_positions is not None and 
        prev_hand_positions is not None and 
        current_hand_positions.shape == prev_hand_positions.shape):
        
        hand_delta = np.linalg.norm(current_hand_positions - prev_hand_positions, axis=1)
        hand_movement = np.mean(hand_delta) * 1000

    prev_hand_positions = current_hand_positions

    # Combine detections
    if body_movement > jerk_threshold or hand_movement > hand_jerk_threshold:
        jerk_counter += 1
    else:
        jerk_counter = max(0, jerk_counter - 1)

    # Detect seizure if jerks continue
    if jerk_counter > frequency_threshold:
        seizure_alert = True
        cv2.putText(frame, "SEIZURE DETECTED!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, (0, 0, 255), 3)

        # Send email once
        if not alert_sent:
            subject = "⚠️ SEIZURE ALERT"
            body = f"A possible seizure was detected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}."
            if send_email_alert(subject, body):
                alert_sent = True
    else:
        seizure_alert = False
        alert_sent = False
        cv2.putText(frame, "No Seizure Detected", (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

    # Display frame
    cv2.imshow("Seizure Detection", frame)

    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
