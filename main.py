import cv2
from pose_detector import PoseDetector
from meme_manager import MemeManager
from collections import deque


def main():
    print("Initializing Meme Tracker...")

    # Initialize detector and meme manager
    detector = PoseDetector()
    meme_manager = MemeManager(assets_dir="assets")

    # Open webcam (0 is usually the default built-in webcam)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Started webcam. Press 'q' to quit.")

    # For smoothing out detections and avoiding flickering, we will require
    # a state to be detected for a few consecutive frames before switching.
    buffer_size = 5
    state_buffer = deque(maxlen=buffer_size)
    current_stable_state = "none"

    while True:
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)

        # Process frame
        results = detector.process_frame(frame)

        # Determine pose state
        state = detector.determine_pose_state(results, frame.shape)

        # Smoothing logic automatically removes the oldest state if len > 5
        state_buffer.append(state)

        # If all items in buffer are the same, update the stable state
        if len(set(state_buffer)) == 1:
            current_stable_state = state_buffer[0]

        # Display meme based on the stable state
        meme_manager.display_meme(current_stable_state)

        # Draw the state on the webcam feed
        color = (0, 255, 0) if current_stable_state != "none" else (0, 0, 255)
        cv2.putText(frame, f"State: {current_stable_state}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        cv2.putText(frame, "Press 'q' to quit", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),
                    1, cv2.LINE_AA)

        # Show webcam feed
        cv2.imshow('Webcam Feed', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
