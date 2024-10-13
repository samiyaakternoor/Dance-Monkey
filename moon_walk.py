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
    "moonwalk": 0,
}

# Define maximum scores for normalization
max_scores = {
    "moonwalk": 1,
}

# Motion detection parameters
motion_threshold = 5000  # Adjust based on sensitivity
last_frame = None

def analyze_moonwalk(landmarks):
    scores["moonwalk"], moonwalk_feedback = check_moonwalk(landmarks)
    return moonwalk_feedback

def check_moonwalk(landmarks):
    left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

    # Check for sliding motion (moonwalk)
    if (left_ankle.y > right_ankle.y) and (left_hip.y > right_hip.y):
        return 1, "Good moonwalk! Keep sliding smoothly."
    return 0, "Try to slide your feet backward while keeping your upper body still."

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
    st.title("Moonwalk Dance Move Detection")
    st.write("Use your webcam to perform the Moonwalk dance move!")

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
                
                moonwalk_feedback = analyze_moonwalk(last_results)

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
                    f"Moonwalk: {'Good' if scores['moonwalk'] else 'Needs Improvement'}\n"
                    f"Overall Score: {percentage_score:.2f}%\n"
                    "Specific Suggestions:\n"
                    f"- Moonwalk: {moonwalk_feedback}\n"
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
