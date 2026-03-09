import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class PoseDetector:
    def __init__(self):
        # Initialize the PoseLandmarker using the generic tasks API
        base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False)
        self.detector = vision.PoseLandmarker.create_from_options(options)

        # Landmark indices (these match standard MediaPipe Pose)
        self.NOSE = 0
        self.LEFT_WRIST = 15
        self.RIGHT_WRIST = 16
        self.LEFT_INDEX = 19
        self.RIGHT_INDEX = 20
        self.LEFT_SHOULDER = 11
        self.RIGHT_SHOULDER = 12

    def process_frame(self, frame):
        # Convert to RGB, then to MediaPipe Image format
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # Detect poses
        results = self.detector.detect(mp_image)
        return results

    def determine_pose_state(self, results, frame_shape):
        """
        Analyzes the landmarks and returns a string indicating the current tracked pose.
        States: 'thinking', 'aha', 'hands_on_cheeks', 'none'
        """
        if not results.pose_landmarks:
            return "none"
            
        # We only look at the first person detected
        landmarks = results.pose_landmarks[0]
        
        # Get relevant landmarks by index
        nose = landmarks[self.NOSE]
        left_wrist = landmarks[self.LEFT_WRIST]
        right_wrist = landmarks[self.RIGHT_WRIST]
        left_index = landmarks[self.LEFT_INDEX]
        right_index = landmarks[self.RIGHT_INDEX]
        
        # Helper to calculate 2D distance
        def distance(lm1, lm2):
            return np.sqrt((lm1.x - lm2.x)**2 + (lm1.y - lm2.y)**2)

        # 1. Check for "Aha!" - one wrist significantly above the shoulder/head
        # y goes down, so a smaller y is higher
        if left_wrist.visibility > 0.6 and left_wrist.y < nose.y - 0.1:
            return "aha"
        if right_wrist.visibility > 0.6 and right_wrist.y < nose.y - 0.1:
            return "aha"

        # 2. Check for "hands on cheeks" / "surprised"
        # Both wrists relatively close to the nose/cheeks
        dist_left_to_nose = distance(left_wrist, nose)
        dist_right_to_nose = distance(right_wrist, nose)
        if left_wrist.visibility > 0.6 and right_wrist.visibility > 0.6:
            if dist_left_to_nose < 0.2 and dist_right_to_nose < 0.2:
                return "hands_on_cheeks"

        # 3. Check for "thinking"
        # One index finger or wrist near the chin area (slightly below the nose)
        chin_y = nose.y + 0.1
        near_chin_threshold = 0.15
        
        if left_index.visibility > 0.6 and abs(left_index.y - chin_y) < near_chin_threshold and abs(left_index.x - nose.x) < near_chin_threshold:
            return "thinking"
        if right_index.visibility > 0.6 and abs(right_index.y - chin_y) < near_chin_threshold and abs(right_index.x - nose.x) < near_chin_threshold:
            return "thinking"

        return "none"
