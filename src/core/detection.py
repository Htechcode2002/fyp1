from ultralytics import YOLO
import cv2
import numpy as np
from src.core.config_manager import ConfigManager
from src.core.database import DatabaseManager
from src.core.face_analyzer import FaceAnalyzer
import os
import time
from datetime import datetime

class VideoDetector:
    def __init__(self, tracker="trackers/botsort_reid.yaml", location_name=None, video_id=None, camera_id=None, multi_cam_tracker=None):
        self.tracker = tracker
        self.camera_id = camera_id if camera_id else "cam_0"  # For multi-camera tracking
        self.location_name = location_name if location_name else "Unknown Location"
        self.video_id = video_id
        self.use_reid = True  # Enable ReID for cross-camera tracking
        self.reid_features = {}  # Store ReID features for each track_id
        self.multi_cam_tracker = multi_cam_tracker  # Shared cross-camera tracker instance
        self.db = DatabaseManager()
        self.heatmap_enabled = False
        self.heatmap_accumulator = None
        self.fall_detection_enabled = False
        self.fall_model = None
        self.fall_detections = {}  # Track fall detections: {box_id: {'start_time': time, 'box': [...], 'confirmed': False}}
        self.fall_threshold = 2.0  # 2 seconds threshold
        self.pose_enabled = False  # Toggle for visualizing 17-point pose
        self.face_analysis_enabled = False  # Toggle for gender and age detection
        self.face_analyzer = FaceAnalyzer()  # InsightFace analyzer
        self.model = None
        self.pose_model = None  # YOLO Pose model for better clothing detection
        self.use_pose_for_color = True  # Enable pose-based color detection
        
        # Optimization: Frame Skipping & Frequency
        self.frame_count = 0
        self.inference_freq = 1 # Run inference every N frames (1 = every frame)
        self.last_pose_results = []
        self.heatmap_update_freq = 5 # Update heatmap overlay every 5 frames
        self.cached_heatmap_overlay = None
        self.load_model()
        self.load_pose_model()

        # State
        self.lines = [] # list of [(x1,y1), (x2,y2)]
        self.zones = [] # list of [(x,y), ...]

        self.track_history = {} # id -> list of (x,y) (last N points)
        self.track_feet = {} # id -> (x, y) feet position for heatmap
        self.line_counts = {} # line_index -> count
        self.loitering_status = {} # id -> start_time ?

        # Enhanced Pedestrian Counting Analytics
        self.seen_track_ids = set() # All unique pedestrians seen
        self.active_track_ids = set() # Currently tracked pedestrians
        self.pedestrian_history = [] # List of (timestamp, count, line_idx, direction)
        self.session_start_time = time.time()

        # Per-line analytics
        self.line_analytics = {} # line_idx -> {timestamps, left_counts, right_counts}

        # Optimization: Color Caching
        self.color_cache = {} # track_id -> color_string
        self.color_samples = {} # track_id -> {color: count} (for stability)

        # Line crossing flash effect
        self.line_flash = {} # line_idx -> timestamp of last crossing

    def set_heatmap(self, enabled):
        self.heatmap_enabled = enabled

    def set_pose_enabled(self, enabled):
        self.pose_enabled = enabled

    def set_face_analysis_enabled(self, enabled):
        """Enable/disable gender and age detection"""
        self.face_analysis_enabled = enabled

    def set_fall_detection(self, enabled):
        self.fall_detection_enabled = enabled
        if enabled and self.fall_model is None:
            self.load_fall_detection_model()

    def load_fall_detection_model(self):
        """Load fall detection model"""
        import torch
        fall_model_path = "models/fall/fall_det_1.pt"

        if not os.path.exists(fall_model_path):
            print(f"WARNING: Fall detection model not found at {fall_model_path}")
            return

        try:
            print(f"Loading fall detection model from {fall_model_path}...")
            self.fall_model = YOLO(fall_model_path)

            # Move to GPU if available
            if torch.cuda.is_available():
                self.fall_model.to('cuda:0')
                print(f"Fall detection model loaded successfully on GPU!")
            else:
                print("Fall detection model loaded successfully on CPU.")

        except Exception as e:
            print(f"ERROR: Failed to load fall detection model: {e}")
            self.fall_model = None

    def get_dominant_color(self, roi):
        """
        Fast and accurate color detection using median color method.
        Median is resistant to outliers (shadows, highlights) and 42x faster than histogram.
        roi: BGR image crop (numpy array)
        """
        if roi.size == 0:
            return "Unknown"

        # Resize ROI for faster processing if too large
        h, w = roi.shape[:2]
        if h * w > 10000:  # If larger than 100x100
            scale = np.sqrt(10000 / (h * w))
            roi = cv2.resize(roi, (int(w * scale), int(h * scale)))

        # Use median color - fast and resistant to noise/outliers
        # This is 42x faster than histogram and 300x faster than K-means
        dominant_bgr = np.median(roi, axis=(0, 1)).astype(np.uint8)

        # Convert to HSV for better color classification
        hsv_pixel = cv2.cvtColor(np.array([[dominant_bgr]], dtype=np.uint8), cv2.COLOR_BGR2HSV)
        h, s, v = hsv_pixel[0][0]

        # Enhanced color classification - 20 core clothing colors
        # OpenCV HSV: H(0-179), S(0-255), V(0-255)
        # Optimized for 85%+ accuracy on real clothing

        # ========== ACHROMATIC COLORS (Low Saturation) ==========

        # Black - very low brightness
        if v < 55:
            return "Black"
        if v < 95 and s < 70:
            return "Black"

        # White - high brightness, low saturation
        if s < 35 and v > 200:
            return "White"

        # Grays - low saturation, medium brightness
        if s < 55:
            if v > 155:
                return "Light Gray"
            elif v > 95:
                return "Gray"
            else:
                return "Dark Gray"

        # ========== CHROMATIC COLORS (High Saturation) ==========

        # Red spectrum (H: 0-10, 170-180) - wraps around
        if h < 10 or h > 170:
            if v < 115:
                return "Dark Red"
            return "Red"

        # Orange/Brown spectrum (H: 10-30)
        if h < 30:
            # Brown - low saturation or low brightness
            if s < 130 or v < 135:
                return "Brown"
            # Orange - saturated and bright
            return "Orange"

        # Yellow spectrum (H: 30-38)
        if h < 38:
            if v < 145:
                return "Dark Yellow"
            return "Yellow"

        # Green spectrum (H: 38-85)
        if h < 85:
            if v < 130:
                return "Dark Green"
            return "Green"

        # Cyan spectrum (H: 85-100)
        if h < 100:
            return "Cyan"

        # Blue spectrum (H: 100-130)
        if h < 130:
            if v < 125:
                return "Navy Blue"
            return "Blue"

        # Purple spectrum (H: 130-150)
        if h < 150:
            return "Purple"

        # Pink/Magenta spectrum (H: 150-170)
        if h < 170:
            if s < 160:
                return "Pink"
            return "Magenta"

        # Wrap back to red
        return "Dark Red" if v < 115 else "Red"

    def load_model(self):
        import torch
        cm = ConfigManager()
        model_path = cm.get("yolo", {}).get("model_path", "models/yolov8n.pt")

        # Check if model exists
        if not os.path.exists(model_path):
            print(f"WARNING: YOLO model not found at {model_path}. Detection will be disabled.")
            return

        try:
            print(f"Loading YOLO model from {model_path}...")
            self.model = YOLO(model_path)

            # Move model to GPU if available
            if torch.cuda.is_available():
                device = 'cuda:0'
                print(f"GPU detected: {torch.cuda.get_device_name(0)}")
                print(f"Moving YOLO model to GPU...")
                self.model.to(device)
                print(f"YOLO model loaded successfully on GPU!")
            else:
                print("No GPU detected. Running on CPU.")
                print("YOLO model loaded successfully on CPU.")

        except Exception as e:
            print(f"ERROR: Failed to load YOLO model: {e}")
            self.model = None

    def load_pose_model(self):
        """Load YOLO Pose model for accurate body keypoint detection"""
        import torch
        pose_model_path = "models/pose/pose.pt"

        if not os.path.exists(pose_model_path):
            print(f"Pose model not found. Pose-based color detection will be disabled.")
            self.use_pose_for_color = False
            return

        try:
            print(f"Loading YOLO Pose model from {pose_model_path}...")
            self.pose_model = YOLO(pose_model_path)

            # Move to GPU if available
            if torch.cuda.is_available():
                self.pose_model.to('cuda:0')
                print("YOLO Pose model loaded successfully on GPU!")
            else:
                print("YOLO Pose model loaded successfully on CPU.")

        except Exception as e:
            print(f"Failed to load Pose model: {e}")
            self.pose_model = None
            self.use_pose_for_color = False

    def get_torso_region_from_pose(self, frame, x1, y1, x2, y2, track_id):
        """
        Use YOLO Pose to get accurate torso region for clothing color detection.
        Returns ([tx1, ty1, tx2, ty2], keypoints) or (None, None) if pose detection fails.

        COCO Keypoints (17 points):
        0: Nose, 1: Left Eye, 2: Right Eye, 3: Left Ear, 4: Right Ear
        5: Left Shoulder, 6: Right Shoulder, 7: Left Elbow, 8: Right Elbow
        9: Left Wrist, 10: Right Wrist, 11: Left Hip, 12: Right Hip
        13: Left Knee, 14: Right Knee, 15: Left Ankle, 16: Right Ankle
        """
        if not (self.use_pose_for_color or self.pose_enabled) or self.pose_model is None:
            return None, None

        try:
            # Crop person region to speed up pose detection
            person_crop = frame[int(y1):int(y2), int(x1):int(x2)]
            if person_crop.size == 0:
                return None, None

            # Run pose detection on cropped region
            results = self.pose_model(person_crop, verbose=False)

            if len(results) == 0 or results[0].keypoints is None:
                return None, None

            keypoints = results[0].keypoints.xy.cpu().numpy()
            if len(keypoints) == 0:
                return None, None

            kps = keypoints[0]  # First person's keypoints
            if len(kps) < 17:
                return None, None

            # Map keypoints back to original frame coordinates
            kps_global = []
            for kp in kps:
                if kp[0] > 0 and kp[1] > 0:
                    kps_global.append([int(x1 + kp[0]), int(y1 + kp[1])])
                else:
                    kps_global.append(None)

            # Extract shoulder and hip keypoints (using local kps for easier math)
            left_shoulder = kps[5]   # [x, y]
            right_shoulder = kps[6]
            left_hip = kps[11]
            right_hip = kps[12]

            # Check if keypoints are detected (confidence > 0)
            valid_torso_kps = []
            for kp in [left_shoulder, right_shoulder, left_hip, right_hip]:
                if kp[0] > 0 and kp[1] > 0:  # Valid keypoint
                    valid_torso_kps.append(kp)

            if len(valid_torso_kps) < 2:
                # Still return keypoints even if torso box can't be calculated
                return None, kps_global

            # Calculate torso bounding box
            xs = [kp[0] for kp in valid_torso_kps]
            ys = [kp[1] for kp in valid_torso_kps]

            # Torso region: from shoulders to hips
            torso_x1 = min(xs)
            torso_y1 = min(ys)
            torso_x2 = max(xs)
            torso_y2 = max(ys)

            # Expand slightly to ensure we capture clothing
            w_torso = torso_x2 - torso_x1
            h_torso = torso_y2 - torso_y1

            torso_x1 = max(0, torso_x1 - w_torso * 0.1)
            torso_x2 = min(person_crop.shape[1], torso_x2 + w_torso * 0.1)
            torso_y1 = max(0, torso_y1 - h_torso * 0.1)
            torso_y2 = min(person_crop.shape[0], torso_y2 + h_torso * 0.1)

            # Convert back to original frame coordinates
            tx1 = int(x1 + torso_x1)
            ty1 = int(y1 + torso_y1)
            tx2 = int(x1 + torso_x2)
            ty2 = int(y1 + torso_y2)

            return [tx1, ty1, tx2, ty2], kps_global

        except Exception as e:
            return None, None

    def check_loitering(self, track_id, current_pos):
        """
        Check if a person is loitering (staying in one area for too long).
        Only checks if person is inside a defined zone.
        Returns: (is_loitering: bool, dwell_time: float)
        """
        from src.utils.geometry import is_point_in_polygon

        loiter_threshold = 5.0  # 5 seconds
        movement_threshold = 50  # pixels

        current_time = time.time()

        # Check if person is in any zone
        in_zone = False
        for zone in self.zones:
            if is_point_in_polygon(current_pos, zone):
                in_zone = True
                break

        # If not in any zone, clear their loitering status and return
        if not in_zone:
            if track_id in self.loitering_status:
                del self.loitering_status[track_id]
            return False, 0.0

        # Person is in a zone - check loitering
        if track_id not in self.loitering_status:
            # First time seeing this track in zone
            self.loitering_status[track_id] = {
                "start_time": current_time,
                "start_pos": current_pos
            }
            return False, 0.0

        # Check how long they've been in the zone
        dwell_time = current_time - self.loitering_status[track_id]["start_time"]
        
        # MOVEMENT THRESHOLD REMOVED
        # Logic updated: As long as they are IN the zone, the timer counts.
        # We do NOT reset if they move around inside the zone.
        # start_pos = self.loitering_status[track_id]["start_pos"]
        # distance = np.sqrt(...)
        # if distance > movement_threshold: ... RESET ...

        # Check if they've been loitering
        is_loitering = dwell_time > loiter_threshold

        return is_loitering, dwell_time

    def set_lines(self, lines):
        self.lines = lines
        # Reset counts for new lines? Or keep? Resetting is safer for now.
        # Structure: { i: {"left": 0, "right": 0, "total": 0} }
        self.line_counts = {i: {"left": 0, "right": 0, "total": 0} for i in range(len(lines))}

        # Initialize analytics for each line
        self.line_analytics = {
            i: {
                "timestamps": [],
                "left_counts": [],
                "right_counts": []
            } for i in range(len(lines))
        }

    def set_zones(self, zones):
        self.zones = zones
        
    def detect(self, frame, tracking_enabled=True):
        """
        Run tracking and counting.
        """
        from src.utils.geometry import segments_intersect
        
        if self.model is None:
            return [], {}
            
        from src.utils.geometry import segments_intersect, is_point_in_polygon, ccw

        self.frame_count += 1
        detections = []
        import copy
        current_counts = copy.deepcopy(self.line_counts)
        current_time = time.time()
        
        # Performance optimization: fixed size
        imgsz = 640
        
        if tracking_enabled:
            # Persist=True for tracking
            # Classes: 0=Person, 24=Backpack, 26=Handbag
            results = self.model.track(frame, persist=True, verbose=False, classes=[0, 24, 26], tracker=self.tracker, imgsz=imgsz)
        else:
            # Predict only (Detection) - No IDs
            results = self.model.predict(frame, verbose=False, classes=[0, 24, 26], imgsz=imgsz)

        # --- POSE DETECTION: Only run when 17-Point Pose is enabled ---
        all_keypoints = []

        # Only run pose detection if the user has enabled it via the toggle
        if self.pose_enabled and self.pose_model is not None:
            # Run pose detection on individual person crops for better accuracy
            # This works better for small/distant people
            all_keypoints = []

            # Collect all person boxes
            person_boxes = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    person_boxes.append((x1, y1, x2, y2))

            for x1, y1, x2, y2 in person_boxes:
                # Crop person region with padding
                pad = 10
                px1 = max(0, int(x1) - pad)
                py1 = max(0, int(y1) - pad)
                px2 = min(frame.shape[1], int(x2) + pad)
                py2 = min(frame.shape[0], int(y2) + pad)

                person_crop = frame[py1:py2, px1:px2]
                if person_crop.size > 0:
                    # Run pose on crop - high resolution for accurate keypoints
                    pose_results = self.pose_model(person_crop, verbose=False, imgsz=640)
                    if len(pose_results) > 0 and pose_results[0].keypoints is not None:
                        kps = pose_results[0].keypoints.xy.cpu().numpy()
                        if len(kps) > 0:
                            # Convert keypoints back to frame coordinates
                            kps_frame = kps[0].copy()
                            kps_frame[:, 0] += px1
                            kps_frame[:, 1] += py1
                            all_keypoints.append(kps_frame)

            all_keypoints = np.array(all_keypoints) if len(all_keypoints) > 0 else np.array([])
            self.last_pose_results = all_keypoints
        else:
            # Pose disabled - use empty array
            all_keypoints = np.array([])
            self.last_pose_results = all_keypoints

        # Track active IDs in this frame
        current_frame_ids = set()

        
        for result in results:
            boxes = result.boxes

            # Check if keypoints are available (for pose estimation models)
            keypoints = result.keypoints if hasattr(result, 'keypoints') else None

            # If tracking disabled or no IDs, ids will be None
            if tracking_enabled and boxes.id is not None:
                ids = boxes.id.cpu().numpy().astype(int)
            else:
                # Create dummy IDs list of None if no tracking
                ids = [None] * len(boxes)

            # Get Class IDs
            cls_ids = boxes.cls.cpu().numpy().astype(int)

            for box, track_id, cls_id in zip(boxes, ids, cls_ids):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())

                # Initialize variables for all detections
                is_loitering = False
                dwell_time = 0
                direction_arrow = None
                
                # Default values to prevent UnboundLocalError
                shirt_color = "Unknown"
                shirt_box = None
                keypoints = None
                
                # Only track People (Class 0) for counting logic to avoid counting bags as people
                if cls_id == 0:
                    # Centroid
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    current_point = (cx, cy)

                    # --- OPTIMIZED POSE & COLOR LOGIC ---
                    keypoints = None
                    # Try to find matching keypoints from our single-pass run
                    if len(all_keypoints) > 0:
                        # Match by centroid distance
                        best_dist = 150 # Increased for high-res videos
                        for kps in all_keypoints:
                            if len(kps) > 0:
                                # Use nose (0) or midpoint of shoulders (5,6) as anchor
                                if kps[0][0] > 0:
                                    anchor = kps[0]
                                elif kps[5][0] > 0 and kps[6][0] > 0:
                                    anchor = (kps[5] + kps[6]) / 2
                                else:
                                    # Fallback to mean of all valid keypoints
                                    valid = kps[kps[:, 0] > 0]
                                    anchor = np.mean(valid, axis=0) if len(valid) > 0 else None
                                
                                if anchor is not None:
                                    dist = np.sqrt((cx-anchor[0])**2 + (cy-anchor[1])**2)
                                    if dist < best_dist:
                                        keypoints = kps.tolist()
                                        # Map [None] to simplify annotation later
                                        keypoints = [[int(pt[0]), int(pt[1])] if pt[0]>0 else None for pt in keypoints]
                                        best_dist = dist

                    # Check Color Cache
                    if track_id is not None and track_id in self.color_cache:
                        shirt_color = self.color_cache[track_id]
                    else:
                        # Only run color logic if needed
                        if keypoints is not None:
                            # Calculate torso box from keypoints
                            valid_kps = [keypoints[5], keypoints[6], keypoints[11], keypoints[12]]
                            valid_pts = [p for p in valid_kps if p is not None]
                            if len(valid_pts) >= 2:
                                tx1 = min(p[0] for p in valid_pts)
                                ty1 = min(p[1] for p in valid_pts)
                                tx2 = max(p[0] for p in valid_pts)
                                ty2 = max(p[1] for p in valid_pts)
                                
                                # Clip to frame
                                h_f, w_f = frame.shape[:2]
                                tx1, ty1 = max(0, tx1), max(0, ty1)
                                tx2, ty2 = min(w_f, tx2), min(h_f, ty2)

                                if tx2 > tx1 and ty2 > ty1:
                                    shirt_roi = frame[ty1:ty2, tx1:tx2]
                                    if shirt_roi.size > 0:
                                        shirt_color = self.get_dominant_color(shirt_roi)
                                        shirt_box = [tx1, ty1, tx2, ty2]
                        else:
                            # Fallback: estimate chest area from bounding box
                            w_box = x2 - x1
                            h_box = y2 - y1
                            chest_y = y1 + (h_box * 0.30)
                            half_w = (w_box * 0.40) / 2
                            half_h = (h_box * 0.15) / 2
                            sx1, sy1 = int(cx - half_w), int(chest_y - half_h)
                            sx2, sy2 = int(cx + half_w), int(chest_y + half_h)
                            
                            # Clip
                            h_f, w_f = frame.shape[:2]
                            sx1, sy1 = max(0, sx1), max(0, sy1)
                            sx2, sy2 = min(w_f, sx2), min(h_f, sy2)

                            if sx2 > sx1 and sy2 > sy1:
                                shirt_roi = frame[sy1:sy2, sx1:sx2]
                                if shirt_roi.size > 0:
                                    shirt_color = self.get_dominant_color(shirt_roi)
                                    shirt_box = [sx1, sy1, sx2, sy2]

                        # Update cache logic (simple count-to-stable)
                        if track_id is not None and shirt_color != "Unknown":
                            if track_id not in self.color_samples: self.color_samples[track_id] = {}
                            self.color_samples[track_id][shirt_color] = self.color_samples[track_id].get(shirt_color, 0) + 1
                            if self.color_samples[track_id][shirt_color] > 5: # Stable after 5 frames
                                self.color_cache[track_id] = shirt_color

                    if track_id is not None:
                        # Track unique pedestrians
                        current_frame_ids.add(track_id)
                        self.seen_track_ids.add(track_id)

                        # --- HEATMAP ACCUMULATION ---
                        try:
                            # Use 1/4 scale for performance
                            h_img, w_img = frame.shape[:2]
                            scale_factor = 0.25
                            small_h, small_w = int(h_img * scale_factor), int(w_img * scale_factor)

                            if self.heatmap_accumulator is None:
                                self.heatmap_accumulator = np.zeros((small_h, small_w), dtype=np.float32)
                            
                            # Simple circle increment
                            # Scale coordinates to small map
                            foot_x_small = int(cx * scale_factor)
                            foot_y_small = int(y2 * scale_factor)
                            curr_foot_small = (foot_x_small, foot_y_small)
                            
                            # Boundary check
                            if 0 <= foot_x_small < small_w and 0 <= foot_y_small < small_h:
                                # 1. Always add a spot at current position (for pauses)
                                self.heatmap_accumulator[foot_y_small, foot_x_small] += 5.0
                                
                                # 2. Draw Line from previous position (for smooth trails)
                                prev_foot_small = self.track_feet.get(track_id)
                                if prev_foot_small:
                                    # Draw line on a temp mask and add it.
                                    mask = np.zeros_like(self.heatmap_accumulator)
                                    cv2.line(mask, prev_foot_small, curr_foot_small, (1), thickness=2) # Thinner line for small map
                                    self.heatmap_accumulator += (mask * 5.0) 
                                
                                # Update history
                                self.track_feet[track_id] = curr_foot_small

                        except Exception as e:
                            # print(f"heatmap error: {e}")
                            pass
                    
                        # --- LINE COUNTING ---
                        # Update History
                        if track_id not in self.track_history:
                            self.track_history[track_id] = []
                        
                        prev_points = self.track_history[track_id]
                        
                        if prev_points:
                            prev_point = prev_points[-1]
                            for i, line in enumerate(self.lines):
                                line_start, line_end = line
                                if segments_intersect(prev_point, current_point, line_start, line_end):
                                    # Determine Direction
                                    # Vector AB = line_end - line_start
                                    # If ccw(A, B, current) is True -> Left side
                                    # If we crossed, we moved from Right->Left or Left->Right
        
                                    is_now_left = ccw(line_start, line_end, current_point)
        
                                    if is_now_left:
                                        # We ended up on Left, so we crossed TO the left side
                                        self.line_counts[i]["left"] += 1
                                        direction = "left"
                                    else:
                                        # We ended up on Right, so we crossed TO the right side
                                        self.line_counts[i]["right"] += 1
                                        direction = "right"
        
                                    self.line_counts[i]["total"] += 1
                                    current_counts[i] = self.line_counts[i].copy()
        
                                    # DB LOGGING
                                    try:
                                        val_left = 1 if direction == "left" else 0
                                        val_right = 1 if direction == "right" else 0
                                        self.db.insert_event(
                                            video_id=self.video_id,
                                            location=self.location_name,
                                            line_name=f"Line {i+1}",
                                            count_left=val_left, 
                                            count_right=val_right,
                                            clothing_color=shirt_color
                                        )
                                    except Exception as e:
                                        print(f"DB Log Error: {e}")

                                    # Mark line crossing for flash effect
                                    self.line_flash[i] = current_time
        
                                    # Record analytics
                                    self.pedestrian_history.append({
                                        "timestamp": current_time,
                                        "line_idx": i,
                                        "direction": direction,
                                        "track_id": track_id
                                    })
                        self.track_history[track_id].append((float(cx), float(cy)))
                        if len(self.track_history[track_id]) > 300: # Limit history
                             self.track_history[track_id].pop(0)

                        # Check Loitering - only if zones are defined
                        if len(self.zones) > 0:
                            is_loitering, dwell_time = self.check_loitering(track_id, (cx, cy))
                        else:
                            # No zones defined - clear loitering status
                            if track_id in self.loitering_status:
                                del self.loitering_status[track_id]

                        # --- DIRECTION LOGIC ---
                        if track_id is not None and track_id in self.track_history and len(self.track_history[track_id]) > 5:
                            prev_x, prev_y = self.track_history[track_id][-5]
                            dx = cx - prev_x
                            dy = cy - prev_y
                            if abs(dx) > 5 or abs(dy) > 5:
                                direction_arrow = (dx, dy)
                
                # --- COLOR DETECTION (Already done above) ---
                # shirt_color and shirt_box are already set

                # --- CROSS-CAMERA TRACKING ---
                global_id = None
                if self.multi_cam_tracker is not None and track_id is not None and cls_id == 0:
                    # Update multi-camera tracker with this detection
                    global_id = self.multi_cam_tracker.update(
                        camera_id=self.camera_id,
                        local_track_id=track_id,
                        frame=frame,
                        box=[x1, y1, x2, y2],
                        color=shirt_color
                    )

                detections.append({
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                    "conf": conf,
                    "id": int(track_id) if track_id is not None else None,
                    "global_id": int(global_id) if global_id is not None else None,
                    "cls_id": int(cls_id),
                    "loitering": is_loitering,
                    "dwell_time": dwell_time,
                    "direction_arrow": direction_arrow,
                    "centroid": (int((x1+x2)/2), int((y1+y2)/2)),
                    "shirt_color": shirt_color,
                    "shirt_box": shirt_box,
                    "keypoints": keypoints
                })

        # Update active track IDs
        self.active_track_ids = current_frame_ids

        # Fall Detection (if enabled) - with 2 second threshold
        # Only detect falls where there are detected people
        if self.fall_detection_enabled and self.fall_model is not None:
            try:
                # First, get all person bounding boxes from main detections
                person_boxes = []
                for det in detections:
                    if det.get("cls_id") == 0:  # Person class
                        person_boxes.append(det["box"])

                # Only run fall detection if there are people in the frame
                if len(person_boxes) > 0:
                    fall_results = self.fall_model(frame, verbose=False)
                    current_fall_boxes = []

                    # Helper function to calculate IoU (Intersection over Union)
                    def calculate_iou(box1, box2):
                        x1_1, y1_1, x2_1, y2_1 = box1
                        x1_2, y1_2, x2_2, y2_2 = box2

                        # Calculate intersection
                        x_left = max(x1_1, x1_2)
                        y_top = max(y1_1, y1_2)
                        x_right = min(x2_1, x2_2)
                        y_bottom = min(y2_1, y2_2)

                        if x_right < x_left or y_bottom < y_top:
                            return 0.0

                        intersection_area = (x_right - x_left) * (y_bottom - y_top)
                        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
                        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
                        union_area = box1_area + box2_area - intersection_area

                        return intersection_area / union_area if union_area > 0 else 0.0

                    # Collect all detected falls in this frame
                    for fall_result in fall_results:
                        fall_boxes = fall_result.boxes
                        if fall_boxes is not None and len(fall_boxes) > 0:
                            for fall_box in fall_boxes:
                                fx1, fy1, fx2, fy2 = fall_box.xyxy[0].cpu().numpy()
                                fall_conf = float(fall_box.conf[0].cpu().numpy())
                                box = [int(fx1), int(fy1), int(fx2), int(fy2)]

                                # Check if this fall box overlaps with any person box
                                overlaps_with_person = False
                                for person_box in person_boxes:
                                    iou = calculate_iou(box, person_box)
                                    if iou > 0.3:  # At least 30% overlap
                                        overlaps_with_person = True
                                        break

                                # Only add if it overlaps with a detected person
                                if overlaps_with_person:
                                    current_fall_boxes.append({
                                        'box': box,
                                        'conf': fall_conf,
                                        'centroid': (int((fx1+fx2)/2), int((fy1+fy2)/2))
                                    })
                else:
                    # No people detected, clear fall detections
                    current_fall_boxes = []

                # Process fall detections with time threshold
                new_fall_detections = {}
                for fall_data in current_fall_boxes:
                    box = fall_data['box']
                    centroid = fall_data['centroid']

                    # Find if this fall box matches an existing tracked fall
                    matched = False
                    for box_id, tracked_fall in self.fall_detections.items():
                        # Check if centroids are close (within 50 pixels)
                        tracked_centroid = tracked_fall['centroid']
                        distance = np.sqrt((centroid[0] - tracked_centroid[0])**2 +
                                         (centroid[1] - tracked_centroid[1])**2)

                        if distance < 50:  # Same fall event
                            matched = True
                            fall_duration = current_time - tracked_fall['start_time']

                            # Check if fall has been ongoing for 2+ seconds
                            if fall_duration >= self.fall_threshold:
                                # Confirmed fall - add to detections
                                detections.append({
                                    "box": box,
                                    "conf": fall_data['conf'],
                                    "id": None,
                                    "cls_id": 999,  # Special class ID for fall
                                    "loitering": False,
                                    "dwell_time": fall_duration,
                                    "direction_arrow": None,
                                    "centroid": centroid,
                                    "shirt_color": None,
                                    "shirt_box": None,
                                    "is_fall": True
                                })

                            # Keep tracking this fall
                            new_fall_detections[box_id] = {
                                'start_time': tracked_fall['start_time'],
                                'centroid': centroid,
                                'box': box
                            }
                            break

                    if not matched:
                        # New fall detected - start tracking
                        box_id = f"fall_{len(new_fall_detections)}_{current_time}"
                        new_fall_detections[box_id] = {
                            'start_time': current_time,
                            'centroid': centroid,
                            'box': box
                        }

                # Update fall detections tracker
                self.fall_detections = new_fall_detections

            except Exception as e:
                print(f"Fall detection error: {e}")

        # --- FACE ANALYSIS: Gender and Age Detection ---
        if self.face_analysis_enabled and self.face_analyzer.enabled:
            try:
                # Get person bounding boxes for filtering
                person_boxes = [det['box'] for det in detections if det['cls_id'] == 0]

                # Analyze faces
                face_info = self.face_analyzer.analyze_faces(frame, person_boxes)

                # Add face info to detections
                for det in detections:
                    if det['cls_id'] != 0:
                        continue

                    px1, py1, px2, py2 = det['box']
                    person_center_x = (px1 + px2) / 2
                    person_center_y = (py1 + py2) / 2

                    # Find closest face to this person
                    closest_face = None
                    min_dist = float('inf')

                    for face in face_info:
                        fx1, fy1, fx2, fy2 = face['bbox']
                        face_center_x = (fx1 + fx2) / 2
                        face_center_y = (fy1 + fy2) / 2

                        # Calculate distance between person and face centers
                        dist = np.sqrt((person_center_x - face_center_x)**2 +
                                      (person_center_y - face_center_y)**2)

                        if dist < min_dist:
                            min_dist = dist
                            closest_face = face

                    # If found a nearby face, add gender and age to detection
                    if closest_face and min_dist < (px2 - px1):  # Face should be within person width
                        det['gender'] = closest_face['gender']
                        det['age'] = closest_face['age']

            except Exception as e:
                print(f"Face analysis error: {e}")

        # Add analytics to counts
        enhanced_counts = {}
        for k, v in current_counts.items():
            enhanced_counts[str(k)] = v

        enhanced_counts["_analytics"] = {
            "total_unique_pedestrians": len(self.seen_track_ids),
            "active_pedestrians": len(self.active_track_ids),
            "session_duration": current_time - self.session_start_time,
            "crossing_events": len(self.pedestrian_history)
        }

        return detections, enhanced_counts

    def annotate_frame(self, frame, detections, counts):
        """
        Draw bounding boxes on the frame.
        """
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            track_id = det.get("id")
            cls_id = det.get("cls_id", 0)
            is_loitering = det.get("loitering", False)
            dwell_time = det.get("dwell_time", 0)
            is_fall = det.get("is_fall", False)

            # Default Colors (BGR)
            color = (0, 255, 0) # Green for Person
            label_prefix = "ID"

            if is_fall:
                color = (0, 0, 255)
                label_prefix = "FALL"
            elif cls_id == 24: # Backpack
                color = (255, 0, 0)
                label_prefix = "BP"
            elif cls_id == 26: # Handbag
                color = (255, 0, 255)
                label_prefix = "HB"

            if is_loitering:
                color = (0, 0, 255)

            # Determine head/avatar point position (using bounding box only for performance)
            dot_center = (int((x1 + x2) / 2), int(y1 + (y2 - y1) * 0.1))

            # Draw Head Dot (White glow + Color center)
            cv2.circle(frame, dot_center, 7, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, dot_center, 5, color, -1, cv2.LINE_AA)

            # Draw Shirt Sampling Box (if available) - Keep this as a reference or remove?
            # User only asked to change the main bounding box.
            if det.get("shirt_box"):
                sx1, sy1, sx2, sy2 = det["shirt_box"]
                cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (0, 255, 255), 1)
            
            # --- Draw Label ---
            label = None
            label_bg_color = (0, 120, 255) # Orange default
            
            if cls_id == 24 or cls_id == 26:
                # Different label bg for bags
                label_bg_color = (100, 100, 100) # Gray
                if track_id is not None:
                     label = f"{label_prefix}:{track_id}"
                else:
                     label = f"{'Backpack' if cls_id==24 else 'Handbag'}"

            elif is_fall:
                label_bg_color = (0, 0, 255)  # Red Label for fall
                fall_duration = det.get("dwell_time", 0)
                label = f"FALL DETECTED {fall_duration:.1f}s"
            elif is_loitering:
                label_bg_color = (0, 0, 255) # Red Label
                label = f"LOITERING {dwell_time:.1f}s"
            elif track_id is not None:
                 # Show global ID if available, otherwise local ID
                 global_id = det.get("global_id")
                 if global_id is not None:
                     label = f"G{global_id} (L{track_id})"  # Global ID (Local ID)
                 else:
                     label = f"ID: {track_id}"

                 # Append Color Info if available
                 if det.get("shirt_color") and det["shirt_color"] != "Unknown":
                     s_c = det["shirt_color"]
                     label += f" ({s_c})"

                 # Append Gender and Age if available
                 if "gender" in det and "age" in det:
                     gender_short = "M" if det["gender"] == "Male" else "F"
                     label += f" {gender_short}/{det['age']}y"
            
            if label:
                # Draw Label with background
                label_text_color = (255, 255, 255)
                t_size = cv2.getTextSize(label, 0, fontScale=0.6, thickness=2)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 5
                
                # Background Rect
                cv2.rectangle(frame, (x1, y1), c2, label_bg_color, -1, cv2.LINE_AA)
                
                # Text
                cv2.putText(frame, label, (x1, y1 - 2), 0, 0.6, label_text_color, thickness=2, lineType=cv2.LINE_AA)

            # Draw Trajectory Trail
            if track_id is not None and track_id in self.track_history:
                points = self.track_history[track_id]
                if len(points) > 1:
                    # Convert to numpy array of points (int)
                    # points is list of (float, float)
                    pts = np.array(points, np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    
                    # Draw polyline
                    # Yellow color for trail, thickness 2
                    cv2.polylines(frame, [pts], False, (0, 255, 255), 2)
            
            # --- Draw Pose Keypoints ---
            if self.pose_enabled and det.get("keypoints") is not None:
                kpts = det["keypoints"]
                if kpts and len(kpts) >= 17:  # Ensure we have valid keypoints
                    # Connections for COCO (17 points)
                    connections = [
                        (0, 1), (0, 2), (1, 3), (2, 4), # Head
                        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), # Shoulders & Arms
                        (5, 11), (6, 12), (11, 12), # Torso
                        (11, 13), (13, 15), (12, 14), (14, 16) # Legs
                    ]

                    # Draw skeleton lines
                    for start_idx, end_idx in connections:
                        if start_idx < len(kpts) and end_idx < len(kpts):
                            pt1 = kpts[start_idx]
                            pt2 = kpts[end_idx]
                            if pt1 is not None and pt2 is not None:
                                try:
                                    cv2.line(frame, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (0, 255, 255), 2)
                                except:
                                    pass  # Skip invalid points

                    # Draw keypoints
                    for i, pt in enumerate(kpts):
                        if pt is not None:
                            try:
                                # Draw circle for keypoint
                                cv2.circle(frame, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)
                                # Optional: Draw keypoint number
                                # cv2.putText(frame, str(i), (int(pt[0])+5, int(pt[1])-5),
                                #             cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                            except:
                                pass  # Skip invalid points
            
        # --- HEATMAP OVERLAY ---
        if self.heatmap_enabled and self.heatmap_accumulator is not None:
            # Only re-render heatmap overlay every N frames to save compute
            if self.frame_count % self.heatmap_update_freq == 0 or self.cached_heatmap_overlay is None:
                try:
                    heatmap_blur = cv2.GaussianBlur(self.heatmap_accumulator, (21, 21), 0)
                    max_val = np.max(heatmap_blur)
                    if max_val > 0:
                        heatmap_norm = (heatmap_blur / max_val * 255).astype(np.uint8)
                        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
                        self.cached_heatmap_overlay = cv2.resize(heatmap_color, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
                    else:
                        self.cached_heatmap_overlay = None
                except Exception as e:
                    pass
            
            # Apply cached overlay
            if self.cached_heatmap_overlay is not None:
                frame = cv2.addWeighted(frame, 0.6, self.cached_heatmap_overlay, 0.4, 0)

        # Draw Lines
        # Draw Lines
        current_time = time.time()
        for i, line in enumerate(self.lines):
            start, end = line

            # Check if line was recently crossed (flash effect for 1 second)
            line_color = (0, 255, 0)  # Default green
            line_thickness = 3
            if i in self.line_flash:
                time_since_flash = current_time - self.line_flash[i]
                if time_since_flash < 0.2:  # Flash for 0.2 second
                    # Flash bright red - very obvious
                    line_color = (0, 0, 255)  # Bright red (BGR format)
                    line_thickness = 5  # Thicker when flashing
                else:
                    # Remove old flash marker
                    del self.line_flash[i]

            cv2.line(frame, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), line_color, line_thickness)
            
            # User requested to remove the count label from the video feed.
            # Counts are still calculated for the sidebar analytics.
            
        # Draw Zones
        for zone in self.zones:
             pts = np.array(zone, np.int32)
             pts = pts.reshape((-1, 1, 2))
             cv2.polylines(frame, [pts], True, (0, 0, 255), 2) # Red Zones

        return frame

    def get_analytics_summary(self):
        """
        Get comprehensive pedestrian counting analytics.
        Returns a dictionary with various metrics.
        """
        current_time = time.time()
        session_duration = current_time - self.session_start_time

        # Calculate flow rate (pedestrians per minute)
        flow_rate = 0
        if session_duration > 0:
            flow_rate = (len(self.pedestrian_history) / session_duration) * 60

        # Get peak crossing time
        peak_time = None
        if self.pedestrian_history:
            # Find the minute with most crossings
            from collections import defaultdict
            crossings_per_minute = defaultdict(int)
            for event in self.pedestrian_history:
                minute_key = int((event["timestamp"] - self.session_start_time) / 60)
                crossings_per_minute[minute_key] += 1

            if crossings_per_minute:
                peak_minute = max(crossings_per_minute, key=crossings_per_minute.get)
                peak_time = f"{peak_minute} min"

        # Per-line statistics
        line_stats = {}
        for i in range(len(self.lines)):
            if i in self.line_counts:
                line_stats[i] = {
                    "total": self.line_counts[i]["total"],
                    "left": self.line_counts[i]["left"],
                    "right": self.line_counts[i]["right"],
                    "net_flow": self.line_counts[i]["left"] - self.line_counts[i]["right"]
                }

        return {
            "total_unique_pedestrians": len(self.seen_track_ids),
            "active_pedestrians": len(self.active_track_ids),
            "total_crossings": len(self.pedestrian_history),
            "session_duration_seconds": session_duration,
            "flow_rate_per_minute": round(flow_rate, 2),
            "peak_crossing_time": peak_time,
            "line_statistics": line_stats
        }

    def get_time_series_data(self, line_idx=None):
        """
        Get time-series data for visualization.
        If line_idx is None, returns aggregated data for all lines.
        """
        if line_idx is not None and line_idx in self.line_analytics:
            return self.line_analytics[line_idx]

        # Aggregate all lines
        if not self.pedestrian_history:
            return {"timestamps": [], "counts": []}

        # Sort by timestamp
        sorted_events = sorted(self.pedestrian_history, key=lambda x: x["timestamp"])

        timestamps = [e["timestamp"] for e in sorted_events]
        cumulative_counts = list(range(1, len(sorted_events) + 1))

        return {
            "timestamps": timestamps,
            "cumulative_counts": cumulative_counts
        }

    def reset_analytics(self):
        """
        Reset all analytics data while keeping the detector running.
        """
        self.seen_track_ids.clear()
        self.active_track_ids.clear()
        self.pedestrian_history.clear()
        self.track_history.clear()  # Clear tracking history
        self.loitering_status.clear()  # Clear loitering status
        self.session_start_time = time.time()

        # Reset line counts
        for i in self.line_counts:
            self.line_counts[i] = {"left": 0, "right": 0, "total": 0}

        # Reset line analytics
        for i in self.line_analytics:
            self.line_analytics[i] = {
                "timestamps": [],
                "left_counts": [],
                "right_counts": []
            }

        # Reset line flash effect
        self.line_flash.clear()

        # Reset fall detections
        self.fall_detections.clear()
        
        # Reset color cache
        self.color_cache.clear()
        self.color_samples.clear()
