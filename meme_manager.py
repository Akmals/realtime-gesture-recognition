import cv2
import os
import numpy as np

class MemeManager:
    def __init__(self, assets_dir="assets"):
        self.assets_dir = assets_dir
        self.memes = {}
        self.default_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(self.default_img, "Waiting for pose...", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Ensure assets directory exists
        if not os.path.exists(self.assets_dir):
            os.makedirs(self.assets_dir)
            
        self.load_memes()
        cv2.namedWindow("Meme Reaction", cv2.WINDOW_NORMAL)
        cv2.imshow("Meme Reaction", self.default_img)
        
    def load_memes(self):
        # We try to load files that the user might have saved
        expected_files = {
            "thinking": ["thinking.jpg", "thinking.png", "thinking_monkey.jpg", "thinking_cat.jpg", "image1.jpg", "image2.jpg", "1.jpg"],
            "aha": ["aha.jpg", "aha.png", "aha_monkey.jpg", "image5.jpg", "5.jpg"],
            "hands_on_cheeks": ["cheeks.jpg", "cheeks.png", "surprised_cat.jpg", "image3.jpg", "3.jpg"]
        }
        
        files_in_dir = os.listdir(self.assets_dir) if os.path.exists(self.assets_dir) else []
        print(f"Files found in assets dir: {files_in_dir}")
        
        for state, filenames in expected_files.items():
            for filename in filenames:
                path = os.path.join(self.assets_dir, filename)
                if os.path.exists(path):
                    img = cv2.imread(path)
                    if img is not None:
                        # Resize for consistent display
                        img = cv2.resize(img, (640, 480))
                        self.memes[state] = img
                        print(f"Successfully loaded meme for state '{state}': {filename}")
                        break
        
        if not self.memes:
            print(f"Warning: No meme images found in {self.assets_dir}.")
            print("Please add some images to the 'assets' folder with names like 'thinking.jpg' or 'aha.jpg'.")

    def display_meme(self, state):
        if state == "none":
            cv2.imshow("Meme Reaction", self.default_img)
        elif state in self.memes:
            cv2.imshow("Meme Reaction", self.memes[state])
        else:
            # If state is recognized but we don't have an image for it, show placeholder with text
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, f"Pose match: {state}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(placeholder, "No image found in assets!", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("Meme Reaction", placeholder)
