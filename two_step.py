import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Score initialization
scores = {
    "posture": 0,
    "footwork": 0,
    "arm_movement": 0,
}

# Define maximum scores for normalization
max_scores = {
    "posture": 1,
    "footwork": 1,
    "arm_movement": 1,
}

# Motion detection parameters
motion_threshold = 5000  # Adjust based on sensitivity
last_frame = None

def analyze_movement(landmarks):
    scores["posture"], posture_feedback = check_posture(landmarks)
    scores["footwork"], footwork_feedback = check_footwork(landmarks)
    scores["arm_movement"], arm_movement_feedback = check_arm_movement(landmarks)

    return posture_feedback, footwork_feedback, arm_movement_feedback

def check_posture(landmarks):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    
    if left_shoulder.y < right_shoulder.y:
        return 1, "Good posture! Keep your shoulders relaxed."
    return 0, "Try to keep your shoulders level and engage your core."

def check_footwork(landmarks):
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
    
    if abs(left_ankle.y - right_ankle.y) < 0.1:  # Example condition for smoothness
        return 1, "Footwork is smooth! Keep it up."
    return 0, "Increase the speed of your steps and make them smoother."

def check_arm_movement(landmarks):
    left_arm = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    right_arm = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    
    if left_arm.y < left_arm.x and right_arm.y < right_arm.x:
        return 1, "Good arm movement! Let your arms sway gently to the music."
    return 0, "Raise your arms higher and allow them to move more freely."

def detect_motion(current_frame):
    global last_frame
    if last_frame is None:
        last_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        return False

    gray_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    diff_frame = cv2.absdiff(last_frame, gray_frame)
    _, thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)
    motion_count = np.sum(thresh_frame > 0)

    last_frame = gray_frame  # Update last frame
    return motion_count > motion_threshold

def main():
    st.title("Dance Move Detection with Motion Detection")
    st.write("Use your webcam to perform dance moves!")

    start_button = st.button("Start Dance")
    stop_button = st.button("Stop Dance")

    # Create a placeholder for video and feedback
    video_placeholder = st.empty()
    report_placeholder = st.empty()

    if start_button:
        cap = cv2.VideoCapture(0)
        last_results = None

        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture video.")
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                last_results = results.pose_landmarks.landmark
                
                # Draw landmarks on the frame
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                
                posture_feedback, footwork_feedback, arm_movement_feedback = analyze_movement(last_results)

                # Detect motion
                if detect_motion(frame):
                    motion_feedback = "Motion detected! Keep moving."
                else:
                    motion_feedback = "No significant motion detected."

                # Display video frame
                video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

                # Calculate overall score
                total_score = sum(scores.values())
                max_total_score = sum(max_scores.values())
                percentage_score = (total_score / max_total_score) * 100

                # Generate report
                report = (
                    "Movement Report:\n"
                    f"Posture: {'Good' if scores['posture'] else 'Needs Improvement'}\n"
                    f"Footwork: {'Good' if scores['footwork'] else 'Needs Improvement'}\n"
                    f"Arm Movement: {'Good' if scores['arm_movement'] else 'Needs Improvement'}\n"
                    f"Overall Score: {percentage_score:.2f}%\n"
                    "Specific Suggestions:\n"
                    f"- Posture: {check_posture(last_results)[1]}\n"
                    f"- Footwork: {check_footwork(last_results)[1]}\n"
                    f"- Arm Movement: {check_arm_movement(last_results)[1]}\n"
                    f"- Motion: {motion_feedback}"
                )
                report_placeholder.text(report)
            else:
                report_placeholder.text("No movements detected.")

            if stop_button:
                break

        cap.release()
        cv2.destroyAllWindows()
    else:
        video_placeholder.text("Click 'Start Dance' to begin.")

    # If needed, cleanup after exiting the loop
    if stop_button:
        st.write("Dance session stopped.")

if __name__ == "__main__":
    main()
