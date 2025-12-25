"""
Face Analysis using InsightFace

Detects faces and analyzes gender and age from video frames.
"""

import cv2
import numpy as np

try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    print("InsightFace not installed. Gender and age detection will be disabled.")
    print("Install with: pip install insightface onnxruntime-gpu")


class FaceAnalyzer:
    """
    Face analyzer using InsightFace for gender and age detection.
    Singleton pattern to ensure models are only loaded once.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FaceAnalyzer, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize InsightFace face analysis model."""
        if getattr(self, '_initialized', False):
            return

        self.app = None
        self.enabled = False
        self.error_message = None

        if not INSIGHTFACE_AVAILABLE:
            self.error_message = "InsightFace library not installed."
            return

        try:
            print("Loading InsightFace model (buffalo_l)...")
            # Initialize face analysis with buffalo_l model (best accuracy)
            self.app = FaceAnalysis(
                name='buffalo_l',
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            self.enabled = True
            self._initialized = True
            print("InsightFace model loaded successfully!")
        except Exception as e:
            error_str = str(e)
            if "getaddrinfo failed" in error_str or "NameResolutionError" in error_str:
                self.error_message = "Network error: Cannot download models from GitHub. Please check internet or download manually."
                print(f"Failed to load InsightFace: {self.error_message}")
            else:
                self.error_message = f"Model load error: {e}"
                print(f"Failed to load InsightFace: {e}")
            
            print("Install models manually if offline: https://github.com/deepinsight/insightface/releases")
            self.enabled = False
            self._initialized = True # Mark as initialized even if failed to prevent repeated attempts

    def analyze_faces(self, frame, person_boxes=None):
        """
        Analyze faces in the frame and return gender and age information.

        Args:
            frame: BGR image frame
            person_boxes: List of person bounding boxes [(x1,y1,x2,y2), ...]
                         If provided, only analyze faces within these boxes

        Returns:
            List of face info dicts: [
                {
                    'bbox': [x1, y1, x2, y2],
                    'gender': 'Male' or 'Female',
                    'age': int (estimated age)
                },
                ...
            ]
        """
        if not self.enabled or self.app is None:
            return []

        try:
            # Detect and analyze faces
            faces = self.app.get(frame)

            results = []
            for face in faces:
                # Get bounding box
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox

                # Get gender (0: Female, 1: Male)
                gender = 'Male' if face.gender == 1 else 'Female'

                # Get age
                age = int(face.age)

                # If person_boxes provided, check if face is within any person box
                if person_boxes is not None:
                    face_center_x = (x1 + x2) / 2
                    face_center_y = (y1 + y2) / 2

                    # Check if face center is within any person box
                    within_person = False
                    for px1, py1, px2, py2 in person_boxes:
                        if px1 <= face_center_x <= px2 and py1 <= face_center_y <= py2:
                            within_person = True
                            break

                    if not within_person:
                        continue  # Skip faces not within person boxes

                results.append({
                    'bbox': [x1, y1, x2, y2],
                    'gender': gender,
                    'age': age
                })

            return results

        except Exception as e:
            print(f"Error analyzing faces: {e}")
            return []

    def draw_face_info(self, frame, face_info):
        """
        Draw face bounding boxes with gender and age labels.

        Args:
            frame: BGR image frame
            face_info: List of face info dicts from analyze_faces()

        Returns:
            Modified frame with face annotations
        """
        for face in face_info:
            x1, y1, x2, y2 = face['bbox']
            gender = face['gender']
            age = face['age']

            # Choose color based on gender
            color = (255, 0, 255) if gender == 'Male' else (255, 192, 203)  # Purple for Male, Pink for Female

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Prepare label
            label = f"{gender}, {age}y"

            # Get text size
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            # Draw label background
            cv2.rectangle(frame,
                         (x1, y1 - text_height - 10),
                         (x1 + text_width + 10, y1),
                         color, -1)

            # Draw text
            cv2.putText(frame, label, (x1 + 5, y1 - 5),
                       font, font_scale, (255, 255, 255), thickness)

        return frame
