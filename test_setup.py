import cv2
import numpy as np
import mediapipe as mp
from pose_detector import PoseDetector

def test():
    print("Testing cv2 version:", cv2.__version__)
    print("Testing numpy version:", np.__version__)
    print("Testing mediapipe version:", mp.__version__)
    
    print("\nInitializing PoseDetector...")
    try:
        detector = PoseDetector()
        print("PoseDetector initialized successfully!")
    except Exception as e:
        print("Failed to initialize PoseDetector:", e)
        return

    # Create a blank image (dummy frame) to test process_frame
    print("\nTesting process_frame with a dummy numpy image (black frame) instead of webcam...")
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    try:
        results = detector.process_frame(dummy_frame)
        print("process_frame ran successfully! (No poses should be detected in a black frame)")
        
        state = detector.determine_pose_state(results, dummy_frame.shape)
        print(f"Detected state for dummy frame: {state}")
        print("\nAll systems GO! You can test pictures or videos without the webcam.")
    except Exception as e:
        print("Failed during process_frame or determine_pose_state:", e)

if __name__ == "__main__":
    test()
