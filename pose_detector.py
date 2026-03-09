import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class PoseDetector:
    def __init__(self):
        # Initialize the PoseLandmarker using the generic tasks API
        base_options = python.BaseOptions(
            model_asset_path='pose_landmarker_lite.task'
            # We explicitly leave out the delegate configuration, letting TFLite
            # fallback to CPU since Mac's Metal GPU backend throws NORM_RECT error
            # when calculating specific facial landmarks without a square projection matrix.
        )
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5)
            
        self.detector = vision.PoseLandmarker.create_from_options(options)

        # Landmark indices (these match standard MediaPipe Pose)
        self.NOSE = 0
        self.LEFT_EYE = 2
        self.RIGHT_EYE = 5
        self.MOUTH_LEFT = 9
        self.MOUTH_RIGHT = 10
        self.LEFT_SHOULDER = 11
        self.RIGHT_SHOULDER = 12
        self.LEFT_WRIST = 15
        self.RIGHT_WRIST = 16
        self.LEFT_INDEX = 19
        self.RIGHT_INDEX = 20

    def process_frame(self, frame):
        # Convert to RGB, then to MediaPipe Image format
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        # Detect poses
        results = self.detector.detect(mp_image)
        return results

    def determine_pose_state(self, results, frame_shape):
        """
        Analyzes the landmarks and returns a string indicating the
        current tracked pose.
        States: 'thinking', 'aha', 'ponder', 'what', 'hehe', 'none'
        """
        if not results.pose_landmarks:
            return "none"

        # We only look at the first person detected
        landmarks = results.pose_landmarks[0]

        # Get relevant landmarks by index
        nose = landmarks[self.NOSE]
        left_eye = landmarks[self.LEFT_EYE]
        right_eye = landmarks[self.RIGHT_EYE]
        mouth_left = landmarks[self.MOUTH_LEFT]
        mouth_right = landmarks[self.MOUTH_RIGHT]
        left_wrist = landmarks[self.LEFT_WRIST]
        right_wrist = landmarks[self.RIGHT_WRIST]
        left_index = landmarks[self.LEFT_INDEX]
        right_index = landmarks[self.RIGHT_INDEX]

        # Helper to calculate 2D distance
        def distance(lm1, lm2):
            return np.sqrt((lm1.x - lm2.x)**2 + (lm1.y - lm2.y)**2)

        class Point:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        # 1. Check for "aha" - finger in the hair (index above eye level)
        if left_index.visibility > 0.6 and left_index.y < left_eye.y - 0.05:
            return "aha"
        if right_index.visibility > 0.6 and right_index.y < right_eye.y - 0.05:
            return "aha"

        # 2. Check for "thinking" (finger to chin) & "ponder" (fist to chin)
        chin_y = nose.y + 0.1
        near_chin_threshold = 0.15
        chin = Point(nose.x, chin_y)

        dist_left_index = distance(left_index, chin)
        dist_right_index = distance(right_index, chin)
        dist_left_wrist = distance(left_wrist, chin)
        dist_right_wrist = distance(right_wrist, chin)

        # Distance between index and wrist to determine if hand is
        # closed (fist)
        left_hand_is_fist = distance(left_wrist, left_index) < 0.06
        right_hand_is_fist = distance(right_wrist, right_index) < 0.06

        if (left_wrist.visibility > 0.6 and
                dist_left_wrist < near_chin_threshold + 0.05):
            if left_hand_is_fist:
                return "ponder"
            elif dist_left_index < near_chin_threshold:
                return "thinking"

        if (right_wrist.visibility > 0.6 and
                dist_right_wrist < near_chin_threshold + 0.05):
            if right_hand_is_fist:
                return "ponder"
            elif dist_right_index < near_chin_threshold:
                return "thinking"

        # 3. Check for "hehe" (sly smile) vs "what" (straight face)
        mouth_dist = distance(mouth_left, mouth_right)
        eye_dist = distance(left_eye, right_eye)

        if eye_dist > 0:
            smile_ratio = mouth_dist / eye_dist
        else:
            smile_ratio = 1.0

        # We assume a wider mouth relative to eye distance indicates a smile
        if smile_ratio > 1.25:
            return "hehe"
        else:
            return "what"
