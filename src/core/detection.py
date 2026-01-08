from ultralytics import YOLO
import cv2
import numpy as np
from src.core.config_manager import ConfigManager
from src.core.database import DatabaseManager
from src.core.face_analyzer import FaceAnalyzer
from src.core.telegram_notifier import create_telegram_notifier
from src.core.model_pool import get_model_pool
from src.core.alert_manager import get_alert_manager
import os
import time
from datetime import datetime

class VideoDetector:
    def __init__(self, tracker="trackers/botsort_reid.yaml", location_name=None, video_id=None, camera_id=None, multi_cam_tracker=None, danger_threshold=100, loitering_threshold=5.0, fall_threshold=2.0):
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
        self.fall_detection_enabled = True  # Default ON
        self.fall_model = None
        self.fall_detections = {}  # Track fall detections: {box_id: {'start_time': time, 'box': [...], 'confirmed': False}}
        self.fall_threshold = fall_threshold  # Configurable fall detection threshold (seconds)

        # OPTIMIZATION: Use global alert manager instead of local cooldown tracking
        # This prevents duplicate alerts across multiple video sources
        self.alert_manager = get_alert_manager()

        # DEPRECATED: Keep for backward compatibility but will use alert_manager instead
        self.fall_alerts_sent = {}  # Will be replaced by alert_manager
        self.fall_alert_cooldown = 60.0  # Will be replaced by alert_manager
        self.loitering_threshold = loitering_threshold  # Configurable loitering detection threshold (seconds)
        self.loitering_alerts_sent = {}  # Will be replaced by alert_manager
        self.loitering_alert_cooldown = 60.0  # Will be replaced by alert_manager
        self.face_analysis_enabled = True  # Default ON - Toggle for gender and age detection
        self.face_analyzer = FaceAnalyzer()  # InsightFace analyzer
        self.mask_detection_enabled = True  # Default ON - Toggle for mask detection
        self.mask_model = None  # YOLO model for mask detection
        self.pose_model = None  # YOLO Pose model for accurate clothing region detection
        self.pose_cache = {}  # track_id -> keypoints (cached after first detection)
        self.danger_threshold = danger_threshold  # Danger threshold for crowd warning
        self.danger_warning_active = False  # Track if danger warning is currently active
        self.telegram_notifier = create_telegram_notifier()  # Telegram alert system

        # OPTIMIZATION: Use shared model pool instead of individual model instances
        self.model_pool = get_model_pool()
        self.model = None
        self.display_mode = "dot"  # Display mode: "dot" (head dot only) or "box" (bounding box)

        # Optimization: Frame Skipping & Frequency
        self.frame_count = 0
        self.inference_freq = 2 # Process every 2nd frame for real-time streams (reduces lag by 50%)
        self.heatmap_update_freq = 5 # Update heatmap overlay every 5 frames
        self.cached_heatmap_overlay = None
        self.face_analysis_freq = 30 # Run face analysis every 30 frames (reduced for performance)

        # Prediction caching for skipped frames
        self.last_detections = []
        self.last_enhanced_counts = {}

        # OPTIMIZATION: Lazy loading - Load models on first detect() call instead of __init__
        # This allows multiple VideoThreads to start immediately without blocking
        self._models_loaded = False
        self._models_loading = False

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

        # Optimization: Smart Color Caching (Fast confirmation)
        self.color_cache = {} # track_id -> color_string (final confirmed color)
        self.color_confirmed = {} # track_id -> bool (True if detected once)

        # Face Analysis Caching (1-confirmation) - OPTIMIZED
        self.face_cache = {} # track_id -> {gender, age} (final confirmed)
        self.face_confirmed = {} # track_id -> bool (True if confirmed - stop detecting)

        # Mask Detection Caching (1-confirmation) - OPTIMIZED
        self.mask_cache = {} # track_id -> mask_status_string (final confirmed)
        self.mask_confirmed = {} # track_id -> bool (True if confirmed - stop detecting)

        # Handbag Detection Tracking (1-confirmation caching)
        self.handbag_cache = {} # track_id -> 1 if has handbag, 0 if no handbag (final confirmed)
        self.handbag_confirmed = {} # track_id -> bool (True if confirmed - stop detecting)

        # Line crossing flash effect
        self.line_flash = {} # line_idx -> timestamp of last crossing

    def set_heatmap(self, enabled):
        self.heatmap_enabled = enabled

    def set_face_analysis_enabled(self, enabled):
        """Enable/disable gender and age detection"""
        self.face_analysis_enabled = enabled

    def set_mask_detection(self, enabled):
        """Enable/disable mask detection"""
        self.mask_detection_enabled = enabled
        if enabled and self.mask_model is None:
            self.load_mask_model()

    def set_display_mode(self, mode):
        """Set display mode: 'dot' or 'box'"""
        if mode in ["dot", "box"]:
            self.display_mode = mode

    def set_fall_detection(self, enabled):
        self.fall_detection_enabled = enabled
        if enabled and self.fall_model is None:
            self.load_fall_detection_model()

    def load_fall_detection_model(self):
        """Load fall detection model from shared pool (GPU optimized)"""
        import torch
        fall_model_path = "models/fall/fall_det_1.pt"

        if not os.path.exists(fall_model_path):
            print(f"[DETECTOR] ⚠️ WARNING: Fall detection model not found at {fall_model_path}")
            return

        try:
            # OPTIMIZATION: Get shared fall detection model from pool
            print(f"[DETECTOR] 🔗 Getting shared Fall detection model from pool...")
            self.fall_model = self.model_pool.get_fall_model(fall_model_path)

            if self.fall_model is not None:
                # Move to GPU if available
                if torch.cuda.is_available():
                    if not (hasattr(self.fall_model, 'device') and 'cuda' in str(self.fall_model.device)):
                        self.fall_model.to('cuda:0')
                    print(f"[DETECTOR] ✅ Shared Fall detection model ready on GPU!")
                else:
                    print("[DETECTOR] 💻 Shared Fall detection model ready on CPU.")

        except Exception as e:
            print(f"[DETECTOR] ❌ ERROR: Failed to load fall detection model: {e}")
            self.fall_model = None

    def filter_skin_and_get_clothing(self, roi):
        """
        Remove skin-colored pixels to isolate clothing.
        Returns clothing pixels only (as array of BGR values)
        """
        # Convert to HSV and YCrCb for robust skin detection
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)

        # HSV skin range (covers light to medium skin tones)
        lower_hsv = np.array([0, 15, 50], dtype=np.uint8)
        upper_hsv = np.array([30, 170, 255], dtype=np.uint8)

        # YCrCb skin range (robust across lighting conditions)
        lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
        upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)

        # Create skin masks
        skin_mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
        skin_mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)

        # Combine masks (OR operation)
        skin_mask = cv2.bitwise_or(skin_mask_hsv, skin_mask_ycrcb)

        # Invert to get clothing mask
        clothing_mask = cv2.bitwise_not(skin_mask)

        # Extract clothing pixels only
        clothing_pixels = roi[clothing_mask > 0]

        return clothing_pixels

    def get_dominant_color(self, roi):
        """
        Ultra-accurate color detection using skin filtering.
        Removes skin pixels (face/hands) to get pure clothing color.
        roi: BGR image crop (numpy array)
        """
        if roi.size == 0:
            return "Unknown"

        h, w = roi.shape[:2]

        # Fast path: downsample for speed
        if h * w > 5000:
            scale = np.sqrt(5000 / (h * w))
            roi = cv2.resize(roi, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            h, w = roi.shape[:2]

        # Method 1: Skin filtering (most accurate)
        try:
            clothing_pixels = self.filter_skin_and_get_clothing(roi)

            # Check if we have enough clothing pixels
            if clothing_pixels.size >= 150:  # At least ~50 pixels (3 channels)
                # Use median of clothing pixels only
                dominant_bgr = np.median(clothing_pixels, axis=0).astype(np.uint8)
            else:
                # Fallback: center sampling
                raise ValueError("Not enough clothing pixels")

        except:
            # Method 2: Center-weighted sampling (fallback)
            margin_h = int(h * 0.25)
            margin_w = int(w * 0.25)

            if h > 2*margin_h and w > 2*margin_w:
                roi_center = roi[margin_h:h-margin_h, margin_w:w-margin_w]
                dominant_bgr = np.median(roi_center, axis=(0, 1)).astype(np.uint8)
            else:
                dominant_bgr = np.median(roi, axis=(0, 1)).astype(np.uint8)

        # Convert to HSV for better color classification
        hsv_pixel = cv2.cvtColor(np.array([[dominant_bgr]], dtype=np.uint8), cv2.COLOR_BGR2HSV)
        h, s, v = hsv_pixel[0][0]

        # Simplified and optimized color classification
        # OpenCV HSV: H(0-179), S(0-255), V(0-255)
        # Focus on common clothing colors for better accuracy

        # ========== ACHROMATIC COLORS (Low Saturation) ==========

        # Black - very low brightness
        if v < 70:
            return "Black"

        # White - high brightness, low saturation
        if s < 40 and v > 190:
            return "White"

        # Gray - low saturation, medium brightness
        if s < 60:
            return "Gray"

        # ========== CHROMATIC COLORS (High Saturation) ==========

        # Red (H: 0-10, 170-180) - wraps around
        if h < 10 or h > 165:
            return "Red"

        # Orange/Brown (H: 10-25)
        if h < 25:
            if v < 120 or s < 100:
                return "Brown"
            return "Orange"

        # Yellow (H: 25-40)
        if h < 40:
            return "Yellow"

        # Green (H: 40-85)
        if h < 85:
            return "Green"

        # Blue (H: 85-130)
        if h < 130:
            return "Blue"

        # Purple (H: 130-155)
        if h < 155:
            return "Purple"

        # Pink (H: 155-165)
        if h < 165:
            return "Pink"

        # Fallback to Red
        return "Red"

    def _ensure_models_loaded(self):
        """
        Lazy loading: Load models on first use to prevent blocking during initialization.
        Returns True if models are ready, False if still loading (allows frame to be displayed without detection).
        """
        if self._models_loaded:
            return True  # Models ready

        if self._models_loading:
            return False  # Still loading, skip detection for this frame

        # Start loading in background thread to avoid blocking
        self._models_loading = True

        def _load_models_async():
            print(f"[DETECTOR] 🚀 Lazy loading models for video_id={self.video_id}...")

            # Load all required models
            self.load_model()

            # Load models for features that are enabled by default
            if self.fall_detection_enabled:
                self.load_fall_detection_model()
            if self.mask_detection_enabled:
                self.load_mask_model()

            # Load pose model for improved clothing color detection
            self.load_pose_model()

            self._models_loaded = True
            self._models_loading = False
            print(f"[DETECTOR] ✅ All models loaded for video_id={self.video_id}")

        # Start background loading thread
        import threading
        threading.Thread(target=_load_models_async, daemon=True).start()

        return False  # Not ready yet, will be ready on next frame

    def load_model(self):
        """Load main YOLO model from shared model pool (GPU optimized)"""
        import torch
        cm = ConfigManager()
        model_path = cm.get("yolo", {}).get("model_path", "models/yolov8n.pt")
        print(f"[DETECTOR] 📋 Configured model path: {model_path}")

        # Check if model exists
        if not os.path.exists(model_path):
            print(f"[DETECTOR] ⚠️ WARNING: YOLO model not found at {model_path}. Detection will be disabled.")
            return

        try:
            # OPTIMIZATION: Get shared model from pool instead of loading new instance
            print(f"[DETECTOR] 🔗 Getting shared YOLO model from pool...")
            self.model = self.model_pool.get_main_model(model_path)

            if self.model is not None:
                # Check GPU status
                if torch.cuda.is_available():
                    device = 'cuda:0'
                    print(f"[DETECTOR] 🎮 GPU detected: {torch.cuda.get_device_name(0)}")
                    # Model is already on GPU from pool, just verify
                    if hasattr(self.model, 'device') and 'cuda' in str(self.model.device):
                        print(f"[DETECTOR] ✅ Shared YOLO model ready on GPU!")
                    else:
                        self.model.to(device)
                        print(f"[DETECTOR] ✅ Moved shared YOLO model to GPU!")
                else:
                    print("[DETECTOR] 💻 No GPU detected. Using shared model on CPU.")
            else:
                print("[DETECTOR] ❌ Failed to get model from pool")

        except Exception as e:
            print(f"[DETECTOR] ❌ CRITICAL ERROR: Failed to load YOLO model: {e}")
            import traceback
            traceback.print_exc()
            self.model = None


    def load_mask_model(self):
        """Load YOLO mask detection model from shared pool (GPU optimized)"""
        import torch
        mask_model_path = "models/mask/mask.pt"

        if not os.path.exists(mask_model_path):
            print(f"[DETECTOR] ⚠️ Mask model not found at {mask_model_path}. Mask detection will be disabled.")
            return

        try:
            # OPTIMIZATION: Get shared mask model from pool
            print(f"[DETECTOR] 🔗 Getting shared Mask detection model from pool...")
            self.mask_model = self.model_pool.get_mask_model(mask_model_path)

            if self.mask_model is not None:
                # Move to GPU if available
                if torch.cuda.is_available():
                    if not (hasattr(self.mask_model, 'device') and 'cuda' in str(self.mask_model.device)):
                        self.mask_model.to('cuda:0')
                    print("[DETECTOR] ✅ Shared Mask model ready on GPU!")
                else:
                    print("[DETECTOR] 💻 Shared Mask model ready on CPU.")

        except Exception as e:
            print(f"[DETECTOR] ❌ Failed to load Mask model: {e}")
            self.mask_model = None

    def load_pose_model(self):
        """Load YOLO Pose model from shared pool (GPU optimized)"""
        import torch
        pose_model_path = "models/pose/pose.pt"

        if not os.path.exists(pose_model_path):
            print(f"[DETECTOR] ⚠️ Pose model not found at {pose_model_path}. Skipping pose-based clothing detection.")
            self.pose_model = None
            return

        try:
            # OPTIMIZATION: Get shared pose model from pool
            print(f"[DETECTOR] 🔗 Getting shared Pose model from pool...")
            self.pose_model = self.model_pool.get_pose_model(pose_model_path)

            if self.pose_model is not None:
                # Move to GPU if available
                if torch.cuda.is_available():
                    if not (hasattr(self.pose_model, 'device') and 'cuda' in str(self.pose_model.device)):
                        self.pose_model.to('cuda:0')
                    print("[DETECTOR] ✅ Shared Pose model ready on GPU!")
                else:
                    print("[DETECTOR] 💻 Shared Pose model ready on CPU.")

        except Exception as e:
            print(f"[DETECTOR] ❌ Failed to load Pose model: {e}")
            self.pose_model = None

    def _extract_keypoints_for_person(self, pose_results, px1, py1, px2, py2):
        """Extract keypoints for a specific person bounding box from pose results"""
        if not pose_results or len(pose_results) == 0:
            return None

        for result in pose_results:
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                keypoints_data = result.keypoints.data
                boxes_data = result.boxes.xyxy

                # Find keypoints that belong to this person box
                for kp, box in zip(keypoints_data, boxes_data):
                    bx1, by1, bx2, by2 = box.cpu().numpy()
                    # Check if this pose box overlaps with person box
                    iou = self._calculate_iou([px1, py1, px2, py2], [bx1, by1, bx2, by2])
                    if iou > 0.5:  # Good overlap
                        return kp.cpu().numpy()  # Return keypoints (17, 3) - [x, y, confidence]
        return None

    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def _get_shirt_box_from_pose(self, keypoints, cx, frame_shape):
        """Calculate shirt box using pose keypoints
        COCO keypoints: 0=nose, 5=left_shoulder, 6=right_shoulder, 11=left_hip, 12=right_hip"""
        h_f, w_f = frame_shape[:2]

        # Extract key points: shoulders (5, 6) and hips (11, 12)
        left_shoulder = keypoints[5][:2]  # [x, y]
        right_shoulder = keypoints[6][:2]
        left_hip = keypoints[11][:2]
        right_hip = keypoints[12][:2]

        # Check if keypoints are valid (confidence > 0)
        shoulders_conf = keypoints[5][2] > 0.3 and keypoints[6][2] > 0.3
        hips_conf = keypoints[11][2] > 0.3 and keypoints[12][2] > 0.3

        if not (shoulders_conf and hips_conf):
            return None  # Not enough confidence

        # Calculate torso region from shoulders to hips
        shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
        hip_y = (left_hip[1] + right_hip[1]) / 2

        # Width based on shoulder distance
        shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
        half_w = shoulder_width * 0.35  # 70% of shoulder width

        # Shirt region: from shoulders to hips
        sx1 = int(cx - half_w)
        sy1 = int(shoulder_y)
        sx2 = int(cx + half_w)
        sy2 = int(hip_y)

        # Clip to frame boundaries
        sx1, sy1 = max(0, sx1), max(0, sy1)
        sx2, sy2 = min(w_f, sx2), min(h_f, sy2)

        return [sx1, sy1, sx2, sy2]


    def _draw_pose_skeleton(self, frame, keypoints):
        """Draw pose skeleton on frame using COCO keypoints
        COCO keypoints order: 0-nose, 1-left_eye, 2-right_eye, 3-left_ear, 4-right_ear,
        5-left_shoulder, 6-right_shoulder, 7-left_elbow, 8-right_elbow,
        9-left_wrist, 10-right_wrist, 11-left_hip, 12-right_hip,
        13-left_knee, 14-right_knee, 15-left_ankle, 16-right_ankle"""

        # Define skeleton connections (pairs of keypoint indices)
        skeleton = [
            # Head
            (0, 1), (0, 2), (1, 3), (2, 4),
            # Torso
            (5, 6), (5, 11), (6, 12), (11, 12),
            # Arms
            (5, 7), (7, 9), (6, 8), (8, 10),
            # Legs
            (11, 13), (13, 15), (12, 14), (14, 16)
        ]

        # Color for skeleton (cyan)
        color = (255, 255, 0)

        # Draw connections
        for start_idx, end_idx in skeleton:
            if start_idx < len(keypoints) and end_idx < len(keypoints):
                start_kp = keypoints[start_idx]
                end_kp = keypoints[end_idx]

                # Check if both keypoints have enough confidence
                if start_kp[2] > 0.3 and end_kp[2] > 0.3:
                    start_point = (int(start_kp[0]), int(start_kp[1]))
                    end_point = (int(end_kp[0]), int(end_kp[1]))
                    cv2.line(frame, start_point, end_point, color, 2)

        # Draw keypoints
        for kp in keypoints:
            if kp[2] > 0.3:  # Confidence threshold
                center = (int(kp[0]), int(kp[1]))
                cv2.circle(frame, center, 3, (0, 255, 255), -1)  # Yellow dots

    def check_loitering(self, track_id, current_pos):
        """
        Check if a person is loitering (staying in one area for too long).
        Only checks if person is inside a defined zone.
        Returns: (is_loitering: bool, dwell_time: float)
        """
        from src.utils.geometry import is_point_in_polygon

        loiter_threshold = self.loitering_threshold  # Use configurable threshold
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
        from src.utils.geometry import segments_intersect, is_point_in_polygon, ccw

        # OPTIMIZATION: Lazy load models on first detect() call (non-blocking)
        # Returns False if models are still loading, allowing frame to be displayed
        models_ready = self._ensure_models_loaded()

        if not models_ready or self.model is None:
            # Models not ready yet or failed to load - return empty results
            # Frame will still be displayed, just without detection
            return [], {}

        self.frame_count += 1

        # CRITICAL: GPU Memory Management - Clear CUDA cache periodically
        # This prevents GPU memory from growing unbounded and causing crashes
        if self.frame_count % 500 == 0:  # Every ~17 seconds at 30 FPS
            try:
                import torch
                import gc
                if torch.cuda.is_available():
                    # Get memory stats before cleanup
                    allocated = torch.cuda.memory_allocated() / 1024**2
                    reserved = torch.cuda.memory_reserved() / 1024**2
                    
                    # Force garbage collection first
                    gc.collect()
                    # Clear unused GPU memory
                    torch.cuda.empty_cache()
                    
                    # Log memory usage
                    if allocated > 2000:  # Warning if > 2GB allocated
                        print(f"⚠️ [GPU] High memory usage: {allocated:.0f}MB allocated, {reserved:.0f}MB reserved")
                    else:
                        print(f"[GPU] Memory: {allocated:.0f}MB allocated, {reserved:.0f}MB reserved (cache/gc cleared)")
            except Exception as e:
                pass  # Ignore errors if torch not available or CUDA not working

        # CRITICAL: Monitor cache sizes to detect memory leaks
        if self.frame_count % 300 == 0:  # Check every 300 frames (every 10 seconds at 30 FPS)
            cache_sizes = {
                'color_cache': len(self.color_cache),
                'face_cache': len(self.face_cache),
                'mask_cache': len(self.mask_cache),
                'handbag_cache': len(self.handbag_cache),
                'track_history': len(self.track_history),
                'seen_track_ids': len(self.seen_track_ids)
            }
            total_cache_size = sum(cache_sizes.values())
            if total_cache_size > 500:  # Warning threshold
                print(f"⚠️ WARNING: Large cache detected ({total_cache_size} entries): {cache_sizes}")
                print(f"   Consider checking if video is looping correctly and reset_analytics() is being called")

        # PERFORMANCE OPTIMIZATION: Skip frames to save GPU
        # If this is a skipped frame, return previous results to keep UI alive
        if self.frame_count % self.inference_freq != 0 and self.last_detections:
            return self.last_detections, self.last_enhanced_counts
            
        detections = []
        import copy
        current_counts = copy.deepcopy(self.line_counts)
        current_time = time.time()
        
        # PERFORMANCE OPTIMIZATION: Balanced resolution for real-time RTSP streams
        imgsz = 640  # Reduced from 960 to 640 for faster inference (2x speed improvement)
        # Lower confidence threshold to reduce GPU floating-point precision differences
        conf_threshold = 0.20  # Slightly higher than before for better precision
        
        if tracking_enabled:
            # Persist=True for tracking
            # Classes: 0=Person, 24=Backpack, 26=Handbag
            # half=True uses FP16 precision - much faster and uses 50% less VRAM on RTX 4090
            results = self.model.track(frame, persist=True, verbose=False, classes=[0, 24, 26], tracker=self.tracker, imgsz=imgsz, conf=conf_threshold, half=True)
        else:
            # Predict only (Detection) - No IDs
            results = self.model.predict(frame, verbose=False, classes=[0, 24, 26], imgsz=imgsz, conf=conf_threshold, half=True)

        # Run pose detection for people who don't have cached keypoints yet
        # This helps with accurate clothing region detection
        pose_results = None
        if self.pose_model is not None and tracking_enabled:
            # Collect track_ids that need pose detection
            people_needing_pose = []
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes
                    if hasattr(boxes, 'id') and boxes.id is not None:
                        ids = boxes.id.cpu().numpy().astype(int)
                        cls_ids = boxes.cls.cpu().numpy().astype(int)
                        for tid, cls_id in zip(ids, cls_ids):
                            if cls_id == 0 and tid not in self.pose_cache:  # Person and not cached
                                people_needing_pose.append(tid)

            # Run pose detection if there are people needing it
            if people_needing_pose:
                pose_results = self.pose_model(frame, verbose=False, half=True)
                # print(f"[POSE] Detected keypoints for {len(people_needing_pose)} new people")

        if not results:
             print("[DEBUG] No results objects returned from model.")
        else:
             if len(results[0].boxes) == 0:
                 # print("[DEBUG] Model ran but detected 0 boxes.")
                 pass


        # Track active IDs in this frame
        current_frame_ids = set()

        # === HANDBAG DETECTION: First pass to collect all detections ===
        # OPTIMIZATION: Only detect for people who DON'T have confirmed handbag status
        # Collect person and handbag detections for association
        person_detections = []  # list of {track_id, box, centroid} - only unconfirmed people
        handbag_detections = []  # list of {box, centroid}

        for result in results:
            boxes = result.boxes
            if tracking_enabled and boxes.id is not None:
                ids = boxes.id.cpu().numpy().astype(int)
            else:
                ids = [None] * len(boxes)
            cls_ids = boxes.cls.cpu().numpy().astype(int)

            for box, track_id, cls_id in zip(boxes, ids, cls_ids):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                if cls_id == 0:  # Person
                    # OPTIMIZATION: Only add to detection list if NOT confirmed yet
                    if track_id is not None and not self.handbag_confirmed.get(track_id, False):
                        person_detections.append({
                            'track_id': track_id,
                            'box': (x1, y1, x2, y2),
                            'centroid': (cx, cy)
                        })
                elif cls_id == 26:  # Handbag (class 26 in COCO)
                    # Only collect handbags if we have unconfirmed people
                    handbag_detections.append({
                        'box': (x1, y1, x2, y2),
                        'centroid': (cx, cy)
                    })

        # Associate handbags with nearby persons (ONLY for unconfirmed people)
        # Skip association if no unconfirmed people (performance optimization)
        if person_detections:
            for person in person_detections:
                track_id = person['track_id']
                if track_id is None:
                    continue

                px, py = person['centroid']
                p_x1, p_y1, p_x2, p_y2 = person['box']

                # Check for nearby handbags (within reasonable distance)
                has_handbag = 0
                for handbag in handbag_detections:
                    hx, hy = handbag['centroid']

                    # Calculate distance between person and handbag
                    distance = np.sqrt((px - hx)**2 + (py - hy)**2)

                    # Check if handbag is near person (distance threshold)
                    person_height = p_y2 - p_y1
                    max_distance = person_height * 0.8  # Handbag within 80% of person height

                    if distance < max_distance:
                        has_handbag = 1
                        break

                # Cache the result and CONFIRM - stop future detection for this track_id
                self.handbag_cache[track_id] = has_handbag
                self.handbag_confirmed[track_id] = True  # CONFIRMED - stop detecting
                if has_handbag:
                    print(f"[HANDBAG CONFIRMED] Track ID {track_id}: Has handbag")
                else:
                    print(f"[HANDBAG CONFIRMED] Track ID {track_id}: No handbag")

        for result in results:
            boxes = result.boxes

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
                send_loitering_alert = False

                # Default values to prevent UnboundLocalError
                shirt_color = "Unknown"
                shirt_box = None

                # Only track People (Class 0) for counting logic to avoid counting bags as people
                if cls_id == 0:
                    # Centroid
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    current_point = (cx, cy)

                    # --- SMART COLOR DETECTION WITH CACHING (Pose Skeleton Only) ---
                    # Check if color is already confirmed
                    if track_id is not None and self.color_confirmed.get(track_id, False):
                        # Use cached color - skip detection to save resources
                        shirt_color = self.color_cache.get(track_id, "Unknown")

                        # Calculate shirt box using cached pose keypoints (required)
                        if track_id in self.pose_cache:
                            shirt_box = self._get_shirt_box_from_pose(self.pose_cache[track_id], cx, frame.shape)
                    else:
                        # Need to detect - only if we have pose keypoints
                        keypoints = None
                        if pose_results is not None and track_id is not None:
                            keypoints = self._extract_keypoints_for_person(pose_results, x1, y1, x2, y2)
                            if keypoints is not None:
                                # Cache keypoints for future use
                                self.pose_cache[track_id] = keypoints

                        # Get shirt box using pose keypoints (cached or new)
                        if keypoints is not None:
                            # Use newly detected keypoints
                            shirt_box = self._get_shirt_box_from_pose(keypoints, cx, frame.shape)
                        elif track_id in self.pose_cache:
                            # Use cached keypoints
                            shirt_box = self._get_shirt_box_from_pose(self.pose_cache[track_id], cx, frame.shape)
                        else:
                            # No pose keypoints available - skip color detection
                            shirt_box = None

                        # Only detect color if we have valid shirt box from pose
                        if shirt_box is not None:
                            sx1, sy1, sx2, sy2 = shirt_box
                            if sx2 > sx1 and sy2 > sy1:
                                shirt_roi = frame[sy1:sy2, sx1:sx2]
                                if shirt_roi.size > 0:
                                    shirt_color = self.get_dominant_color(shirt_roi)

                                    # Update color cache for this track_id (only if NOT yet confirmed)
                                    if track_id is not None and shirt_color != "Unknown" and not self.color_confirmed.get(track_id, False):
                                        # Confirmed! Cache and stop detecting immediately
                                        self.color_cache[track_id] = shirt_color
                                        self.color_confirmed[track_id] = True
                                        print(f"[COLOR CONFIRMED] Track ID {track_id}: {shirt_color}")

                    if track_id is not None:
                        # Track unique pedestrians
                        current_frame_ids.add(track_id)
                        self.seen_track_ids.add(track_id)

                        # --- HEATMAP ACCUMULATION ---
                        if self.heatmap_enabled:
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

                                        # Get Gender/Age from cache if available
                                        gender_val = None
                                        age_val = None
                                        if track_id is not None and track_id in self.face_cache:
                                            gender_val = self.face_cache[track_id]['gender']
                                            age_val = self.face_cache[track_id]['age']

                                        # Get Mask Status from cache if available
                                        mask_val = None
                                        if track_id is not None and track_id in self.mask_cache:
                                            mask_val = self.mask_cache[track_id]

                                        # Get Handbag Status from cache if available
                                        handbag_val = 0
                                        if track_id is not None and track_id in self.handbag_cache:
                                            handbag_val = self.handbag_cache[track_id]

                                        print(f"[DEBUG] Line crossing - track_id: {track_id}, mask_val: {mask_val}, handbag: {handbag_val}")

                                        self.db.insert_event(
                                            video_id=self.video_id,
                                            location=self.location_name,
                                            line_name=f"Line {i+1}",
                                            count_left=val_left,
                                            count_right=val_right,
                                            clothing_color=shirt_color,
                                            gender=gender_val,
                                            age=age_val,
                                            mask_status=mask_val,
                                            handbag=handbag_val
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
                        if len(self.track_history[track_id]) > 30: # Limit to last 30 points (~1 second at 30 FPS)
                             self.track_history[track_id].pop(0)

                        # Check Loitering - only if zones are defined
                        if len(self.zones) > 0:
                            is_loitering, dwell_time = self.check_loitering(track_id, (cx, cy))

                            # Check if we should send loitering alert (using global alert manager)
                            if is_loitering and track_id is not None:
                                # Use global alert manager to check cooldown across all video sources
                                if self.alert_manager.can_send_loitering_alert(self.video_id, track_id):
                                    send_loitering_alert = True
                                    print(f"[LOITERING ALERT] ✅ MARKED FOR SEND - video_id={self.video_id}, track_id={track_id}, dwell_time={dwell_time:.1f}s")
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
                    "send_loitering_alert": send_loitering_alert,
                    "direction_arrow": direction_arrow,
                    "centroid": (int((x1+x2)/2), int((y1+y2)/2)),
                    "shirt_color": shirt_color,
                    "shirt_box": shirt_box
                })

        # Update active track IDs
        self.active_track_ids = current_frame_ids

        # Check if current people count exceeds danger threshold
        current_people_count = len(self.active_track_ids)
        if current_people_count > self.danger_threshold:
            self.danger_warning_active = True
        else:
            self.danger_warning_active = False

        # Clean up all caches for IDs that are no longer active
        # Keep cache for recently seen IDs (for re-identification)
        all_tracked_ids = set(self.color_cache.keys()) | set(self.face_cache.keys()) | set(self.mask_cache.keys())
        # FIX: Clean up if not seen in CURRENT frame (or seen_track_ids which grows forever)
        # Actually, let's keep them in cache but limit total cache size
        inactive_ids = all_tracked_ids - current_frame_ids
        
        # Limit cache size to prevent memory leaks
        # If cache > 100 entries, start removing oldest inactive IDs
        if len(inactive_ids) > 100:
            # Sort by ID or just take a slice
            ids_to_remove = list(inactive_ids)[:len(inactive_ids) - 50]
            for tid in ids_to_remove:
                # Color cache
                if tid in self.color_cache:
                    del self.color_cache[tid]
                if tid in self.color_confirmed:
                    del self.color_confirmed[tid]
                # Face cache
                if tid in self.face_cache:
                    del self.face_cache[tid]
                if tid in self.face_confirmed:
                    del self.face_confirmed[tid]
                if tid in self.face_samples:
                    del self.face_samples[tid]
                # Mask cache
                if tid in self.mask_cache:
                    del self.mask_cache[tid]
                if tid in self.mask_confirmed:
                    del self.mask_confirmed[tid]
                if tid in self.mask_samples:
                    del self.mask_samples[tid]
                # Handbag cache
                if tid in self.handbag_cache:
                    del self.handbag_cache[tid]
                if tid in self.handbag_confirmed:
                    del self.handbag_confirmed[tid]
                # Feet history
                if tid in self.track_feet:
                    del self.track_feet[tid]
                # History
                if tid in self.track_history:
                    del self.track_history[tid]
                # ReID/Multi-cam features
                if hasattr(self, 'reid_features') and tid in self.reid_features:
                    del self.reid_features[tid]

        # Fall Detection (if enabled) - with 2 second threshold
        # Only detect falls where there are detected people
        if self.fall_detection_enabled and self.fall_model is not None:
            try:
                # First, get all person bounding boxes AND track_ids from main detections
                person_boxes_with_ids = []
                for det in detections:
                    if det.get("cls_id") == 0:  # Person class
                        person_boxes_with_ids.append({
                            'box': det["box"],
                            'track_id': det.get("id")  # May be None if tracking disabled
                        })

                # Only run fall detection if there are people in the frame
                if len(person_boxes_with_ids) > 0:
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

                    # Collect all detected falls in this frame and match to track_id
                    for fall_result in fall_results:
                        fall_boxes = fall_result.boxes
                        if fall_boxes is not None and len(fall_boxes) > 0:
                            for fall_box in fall_boxes:
                                fx1, fy1, fx2, fy2 = fall_box.xyxy[0].cpu().numpy()
                                fall_conf = float(fall_box.conf[0].cpu().numpy())
                                
                                # OPTIMIZATION: Skip low confidence detections (reduce false positives)
                                if fall_conf < 0.5:  # Increased from default ~0.25
                                    continue
                                
                                box = [int(fx1), int(fy1), int(fx2), int(fy2)]

                                # Find the person this fall belongs to (highest IoU)
                                best_match_track_id = None
                                best_iou = 0.0
                                for person_data in person_boxes_with_ids:
                                    iou = calculate_iou(box, person_data['box'])
                                    if iou > best_iou:
                                        best_iou = iou
                                        best_match_track_id = person_data['track_id']

                                # Only add if overlap is good enough (50%+ to reduce false positives)
                                if best_iou > 0.5:
                                    current_fall_boxes.append({
                                        'box': box,
                                        'conf': fall_conf,
                                        'centroid': (int((fx1+fx2)/2), int((fy1+fy2)/2)),
                                        'track_id': best_match_track_id  # Associate fall with person's track_id
                                    })
                else:
                    # No people detected, clear fall detections
                    current_fall_boxes = []

                # Process fall detections with time threshold and track_id-based cooldown
                new_fall_detections = {}
                fall_alerts_marked_this_frame = set()  # Track which IDs we've already marked for alert in this frame
                for fall_data in current_fall_boxes:
                    box = fall_data['box']
                    centroid = fall_data['centroid']
                    track_id = fall_data['track_id']  # Person's track_id

                    # Use track_id as primary key if available, otherwise fall back to position-based matching
                    if track_id is not None:
                        # Track using person's ID
                        box_id = f"fall_track_{track_id}"

                        # Check if this person already has a tracked fall
                        if box_id in self.fall_detections:
                            # Existing fall for this person
                            tracked_fall = self.fall_detections[box_id]
                            fall_duration = current_time - tracked_fall['start_time']

                            # Check if fall has been ongoing for 2+ seconds
                            if fall_duration >= self.fall_threshold:
                                # Confirmed fall - add to detections
                                detections.append({
                                    "box": box,
                                    "conf": fall_data['conf'],
                                    "id": track_id,
                                    "cls_id": 999,  # Special class ID for fall
                                    "loitering": False,
                                    "dwell_time": fall_duration,
                                    "direction_arrow": None,
                                    "centroid": centroid,
                                    "shirt_color": None,
                                    "shirt_box": None,
                                    "is_fall": True
                                })

                                # Use global alert manager to check cooldown across all video sources
                                if self.alert_manager.can_send_fall_alert(self.video_id, track_id):
                                    # Cooldown period passed or first time - can send alert
                                    detections[-1]['send_fall_alert'] = True
                                    detections[-1]['fall_track_id'] = track_id
                                    # DO NOT update timestamp here - will update in annotate_frame after actual send
                                    print(f"[FALL ALERT] ✅ MARKED FOR SEND - video_id={self.video_id}, track_id={track_id}, duration={fall_duration:.1f}s")

                            # Keep tracking this fall
                            new_fall_detections[box_id] = {
                                'start_time': tracked_fall['start_time'],
                                'centroid': centroid,
                                'box': box,
                                'track_id': track_id
                            }
                        else:
                            # New fall detected for this person - start tracking
                            new_fall_detections[box_id] = {
                                'start_time': current_time,
                                'centroid': centroid,
                                'box': box,
                                'track_id': track_id
                            }
                            # Removed verbose logging to reduce console spam
                    else:
                        # No track_id available - fall back to position-based matching (legacy behavior)
                        matched = False
                        for box_id, tracked_fall in self.fall_detections.items():
                            if not box_id.startswith("fall_track_"):  # Only match position-based falls
                                tracked_centroid = tracked_fall['centroid']
                                distance = np.sqrt((centroid[0] - tracked_centroid[0])**2 +
                                                 (centroid[1] - tracked_centroid[1])**2)

                                if distance < 150:  # Same fall event
                                    matched = True
                                    fall_duration = current_time - tracked_fall['start_time']

                                    if fall_duration >= self.fall_threshold:
                                        detections.append({
                                            "box": box,
                                            "conf": fall_data['conf'],
                                            "id": None,
                                            "cls_id": 999,
                                            "loitering": False,
                                            "dwell_time": fall_duration,
                                            "direction_arrow": None,
                                            "centroid": centroid,
                                            "shirt_color": None,
                                            "shirt_box": None,
                                            "is_fall": True
                                        })

                                    new_fall_detections[box_id] = {
                                        'start_time': tracked_fall['start_time'],
                                        'centroid': centroid,
                                        'box': box,
                                        'track_id': None
                                    }
                                    break

                        if not matched:
                            box_id = f"fall_pos_{len(new_fall_detections)}_{current_time}"
                            new_fall_detections[box_id] = {
                                'start_time': current_time,
                                'centroid': centroid,
                                'box': box,
                                'track_id': None
                            }

                # Update fall detections tracker
                self.fall_detections = new_fall_detections

            except Exception as e:
                print(f"Fall detection error: {e}")

        # --- FACE ANALYSIS: Gender and Age Detection (with 1-confirmation caching) ---
        # CRITICAL OPTIMIZATION: Only run every Nth frame to prevent blocking
        if self.face_analysis_enabled and self.face_analyzer.enabled and self.frame_count % self.face_analysis_freq == 0:
            try:
                # OPTIMIZATION: Only analyze people who don't have CONFIRMED face info!
                people_to_analyze = []
                for det in detections:
                    if det['cls_id'] == 0:
                        tid = det.get('id')
                        # If confirmed, use cached values
                        if tid is not None and self.face_confirmed.get(tid, False):
                            det['gender'] = self.face_cache[tid]['gender']
                            det['age'] = self.face_cache[tid]['age']
                        else:
                            # Need to analyze this person
                            people_to_analyze.append(det)

                if people_to_analyze:
                    # Get person bounding boxes for filtering in FaceAnalyzer
                    person_boxes = [p['box'] for p in people_to_analyze]

                    # Analyze faces ONLY for those who need it
                    face_info = self.face_analyzer.analyze_faces(frame, person_boxes)

                    # Match detected faces back to the people who needed analysis
                    for det in people_to_analyze:
                        px1, py1, px2, py2 = det['box']
                        person_center_x = (px1 + px2) / 2
                        person_center_y = (py1 + py2) / 2
                        tid = det.get('id')

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

                        # FaceAnalyzer already filtered, so if we found a face, it's valid
                        if closest_face and tid is not None and not self.face_confirmed.get(tid, False):
                            gender = closest_face['gender']
                            age = closest_face['age']

                            # ✅ 1-CONFIRMATION LOGIC - Cache immediately, stop detecting
                            self.face_cache[tid] = {'gender': gender, 'age': age}
                            self.face_confirmed[tid] = True
                            det['gender'] = gender
                            det['age'] = age
                            print(f"[FACE CONFIRMED] Track ID {tid}: {gender}, Age {age}")

            except Exception as e:
                print(f"Face analysis error: {e}")
        else:
            # Skipped frame - use cached face info for confirmed people
            if self.face_analysis_enabled and self.face_analyzer.enabled:
                for det in detections:
                    if det['cls_id'] == 0:
                        tid = det.get('id')
                        if tid is not None and self.face_confirmed.get(tid, False):
                            det['gender'] = self.face_cache[tid]['gender']
                            det['age'] = self.face_cache[tid]['age']

        # --- MASK DETECTION (with 1-confirmation caching) ---
        if self.mask_detection_enabled and self.mask_model is not None:
            # print(f"[DEBUG] Mask detection ENABLED and running on frame")
            try:
                # Process each person detection
                for det in detections:
                    if det['cls_id'] == 0:  # Person
                        tid = det.get('id')

                        # Check if already confirmed
                        if tid is not None and self.mask_confirmed.get(tid, False):
                            # Use cached mask status
                            mask_str = self.mask_cache.get(tid, "Unknown")
                            det['mask_status_str'] = mask_str

                            # Also set numeric mask_status for visualization
                            mask_labels_reverse = {"With Mask": 0, "No Mask": 1, "Mask Incorrect": 2}
                            det['mask_status'] = mask_labels_reverse.get(mask_str, None)
                            det['mask_confidence'] = 1.0  # Confirmed
                            det['mask_box'] = None  # Don't need box after confirmation
                            continue  # Skip detection for this person

                # Run mask detection ONLY for unconfirmed people
                people_to_detect = [det for det in detections if det['cls_id'] == 0 and not self.mask_confirmed.get(det.get('id'), False)]

                if people_to_detect:
                    # Run mask detection on the frame
                    mask_results = self.mask_model(frame, verbose=False, half=True)

                    # Extract mask detections
                    mask_detections = []
                    for result in mask_results:
                        if result.boxes is not None:
                            for box in result.boxes:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                conf = float(box.conf[0].cpu().numpy())
                                cls_id = int(box.cls[0].cpu().numpy())

                                mask_detections.append({
                                    'box': (int(x1), int(y1), int(x2), int(y2)),
                                    'confidence': conf,
                                    'class': cls_id  # 0: with_mask, 1: without_mask, 2: mask_weared_incorrect
                                })

                    # Match mask detections to people who need detection
                    for det in people_to_detect:
                        px1, py1, px2, py2 = det['box']
                        person_center_x = (px1 + px2) / 2
                        person_center_y = (py1 + py2) / 2
                        tid = det.get('id')

                        # Find closest mask detection to this person
                        closest_mask = None
                        min_dist = float('inf')

                        for mask_det in mask_detections:
                            mx1, my1, mx2, my2 = mask_det['box']
                            mask_center_x = (mx1 + mx2) / 2
                            mask_center_y = (my1 + my2) / 2

                            # Calculate distance
                            dist = np.sqrt((person_center_x - mask_center_x)**2 +
                                          (person_center_y - mask_center_y)**2)

                            if dist < min_dist:
                                min_dist = dist
                                closest_mask = mask_det

                        # Determine mask status
                        if closest_mask and min_dist < (px2 - px1):  # Within person width
                            mask_class = closest_mask['class']
                            det['mask_status'] = mask_class
                            det['mask_confidence'] = closest_mask['confidence']
                            det['mask_box'] = closest_mask['box']  # Save mask bounding box

                            # Convert to string
                            mask_labels = {0: "With Mask", 1: "No Mask", 2: "Mask Incorrect"}
                            mask_str = mask_labels.get(mask_class, "Unknown")
                            det['mask_status_str'] = mask_str
                        else:
                            # No mask detected for this person in this frame
                            det['mask_status'] = 1  # No Mask
                            det['mask_confidence'] = 0.0
                            det['mask_box'] = None
                            mask_str = "No Mask"
                            det['mask_status_str'] = mask_str

                        # FAST CACHING LOGIC (1-time confirmation)
                        if tid is not None and not self.mask_confirmed.get(tid, False):
                            # CONFIRMED! Cache and stop detecting
                            self.mask_cache[tid] = mask_str
                            self.mask_confirmed[tid] = True
                            print(f"[MASK CONFIRMED] Track ID {tid}: {mask_str}")

            except Exception as e:
                print(f"Mask detection error: {e}")

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

        # Cache results for skipped frames
        self.last_detections = detections
        self.last_enhanced_counts = enhanced_counts

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

            # Display based on mode (always show box for falls)
            if self.display_mode == "box" or is_fall:
                # Draw full bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            else:  # "dot" mode (default)
                # Determine head/avatar point position (using bounding box only for performance)
                dot_center = (int((x1 + x2) / 2), int(y1 + (y2 - y1) * 0.1))

                # Draw Head Dot (White glow + Color center)
                cv2.circle(frame, dot_center, 7, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, dot_center, 5, color, -1, cv2.LINE_AA)

            # Draw Shirt Sampling Box (only during detection, hide after confirmation)
            if det.get("shirt_box") and track_id is not None:
                # Check if color is confirmed
                is_confirmed = self.color_confirmed.get(track_id, False)

                # Only show box if NOT confirmed yet
                if not is_confirmed:
                    sx1, sy1, sx2, sy2 = det["shirt_box"]
                    shirt_color = det.get("shirt_color", "Unknown")

                    # Map color name to BGR values for display
                    color_bgr_map = {
                        "Red": (0, 0, 255),
                        "Green": (0, 255, 0),
                        "Blue": (255, 0, 0),
                        "Yellow": (0, 255, 255),
                        "Orange": (0, 165, 255),
                        "Purple": (255, 0, 255),
                        "Pink": (203, 192, 255),
                        "Brown": (42, 42, 165),
                        "Gray": (128, 128, 128),
                        "Black": (50, 50, 50),
                        "White": (245, 245, 245),
                        "Unknown": (200, 200, 200)
                    }

                    # Get the color for filling the box
                    fill_color = color_bgr_map.get(shirt_color, (200, 200, 200))

                    # Fill the box with semi-transparent detected color
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (sx1, sy1), (sx2, sy2), fill_color, -1)  # Fill with detected color
                    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)  # 30% transparency for subtler display

                    # Draw border - white for detecting
                    border_color = (255, 255, 255)
                    cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), border_color, 2)

                    # Set color label text
                    color_label = f"{shirt_color}"

                    # Get text size
                    (tw, th), _ = cv2.getTextSize(color_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                    # Draw label background - yellow for detecting
                    label_bg_color = (0, 255, 255)
                    cv2.rectangle(frame, (sx1, sy1 - th - 6), (sx1 + tw + 4, sy1), label_bg_color, -1)

                    # Draw label text
                    cv2.putText(frame, color_label, (sx1 + 2, sy1 - 3),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            # Draw Pose Skeleton (only if NOT confirmed yet)
            if track_id is not None and track_id in self.pose_cache:
                is_confirmed = self.color_confirmed.get(track_id, False)
                if not is_confirmed:
                    keypoints = self.pose_cache[track_id]
                    self._draw_pose_skeleton(frame, keypoints)

            # Draw Mask Bounding Box (only if NOT confirmed - during confirmation phase)
            if det.get("mask_box") and track_id is not None:
                is_mask_confirmed = self.mask_confirmed.get(track_id, False)

                # Only show box if NOT confirmed yet
                if not is_mask_confirmed:
                    mx1, my1, mx2, my2 = det["mask_box"]
                    mask_class = det.get("mask_status", -1)

                    # Different colors for different mask statuses
                    if mask_class == 0:
                        mask_color = (0, 255, 0)  # Green for with mask
                    elif mask_class == 1:
                        mask_color = (0, 0, 255)  # Red for no mask
                    elif mask_class == 2:
                        mask_color = (0, 165, 255)  # Orange for incorrect mask
                    else:
                        mask_color = (255, 255, 255)  # White for unknown

                    # Draw mask box with thicker line
                    cv2.rectangle(frame, (mx1, my1), (mx2, my2), mask_color, 2)

                    # Show label on mask box
                    mask_confirm_label = f"{mask_str}"

                    # Get text size
                    (tw, th), _ = cv2.getTextSize(mask_confirm_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)

                    # Draw label background
                    cv2.rectangle(frame, (mx1, my1 - th - 4), (mx1 + tw, my1), (0, 255, 255), -1)

                    # Draw label text
                    cv2.putText(frame, mask_confirm_label, (mx1, my1 - 2),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

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
                     label = f"ID:{global_id}"  # Simplified: just show global ID
                 else:
                     label = f"ID:{track_id}"

                 # Append Gender and Age if available (BEFORE color)
                 if "gender" in det and "age" in det:
                     is_face_confirmed = self.face_confirmed.get(track_id, False)
                     if is_face_confirmed:
                         # Show gender/age in label only after confirmation
                         gender_short = "M" if det["gender"] == "Male" else "F"
                         label += f" {gender_short}/{det['age']}"
                     else:
                         # Show "Calculating..." while confirming
                         face_sample_count = len(self.face_samples.get(track_id, []))
                         if face_sample_count > 0:
                             label += f" Calculating..."

                 # Append Color Info ONLY if confirmed (after 3 detections)
                 if det.get("shirt_color") and det["shirt_color"] != "Unknown":
                     is_confirmed = self.color_confirmed.get(track_id, False)
                     if is_confirmed:
                         # Only show color in label after confirmation
                         s_c = det["shirt_color"]
                         label += f" {s_c}"  # Removed parentheses for cleaner look

                 # Append Mask Status if available
                 if "mask_status" in det and det["mask_status"] is not None:
                     # Show mask status in label
                     mask_class = det["mask_status"]
                     if mask_class == 0:
                         label += " [MASK]"  # With mask
                     elif mask_class == 1:
                         label += " [NO MASK]"  # Without mask
                     elif mask_class == 2:
                         label += " [MASK ERR]"  # Incorrect mask

                 # Append Handbag Status if available
                 if track_id is not None and self.handbag_confirmed.get(track_id, False):
                     has_handbag = self.handbag_cache.get(track_id, 0)
                     if has_handbag == 1:
                         label += " [BAG]"  # Has handbag

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

        # Draw Danger Warning if threshold exceeded
        if self.danger_warning_active:
            h, w = frame.shape[:2]

            # Create semi-transparent red overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

            # Draw red border
            border_thickness = 15
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), border_thickness)

            # Warning text
            current_count = len(self.active_track_ids)
            warning_text = f"!!! DANGER: {current_count} PEOPLE (Limit: {self.danger_threshold})"

            # Calculate text size
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5
            thickness = 4
            (text_w, text_h), baseline = cv2.getTextSize(warning_text, font, font_scale, thickness)

            # Draw text background (red)
            text_x = (w - text_w) // 2
            text_y = 60
            padding = 20
            cv2.rectangle(frame,
                         (text_x - padding, text_y - text_h - padding),
                         (text_x + text_w + padding, text_y + padding),
                         (0, 0, 255), -1)

            # Draw white text
            cv2.putText(frame, warning_text, (text_x, text_y),
                       font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

            # Send Telegram alert with screenshot (using global cooldown)
            if self.telegram_notifier and self.telegram_notifier.enabled:
                # Check global cooldown for danger alerts
                if self.alert_manager.can_send_danger_alert(self.video_id):
                    self.telegram_notifier.send_danger_alert(
                        frame=frame,
                        current_count=current_count,
                        threshold=self.danger_threshold,
                        location=self.location_name
                    )
                    # Mark danger alert as sent
                    self.alert_manager.mark_danger_alert_sent(self.video_id)
                    print(f"[DANGER ALERT] ✅ Sent for video_id={self.video_id}, count={current_count}")

        # Check for fall alerts to send (after all visualization is complete)
        # DEBUG: Count how many fall alerts are marked
        fall_alerts_to_send = [d for d in detections if d.get('send_fall_alert', False)]
        if fall_alerts_to_send:
            print(f"\n[TELEGRAM DEBUG] Found {len(fall_alerts_to_send)} detections marked for alert")
            for idx, d in enumerate(fall_alerts_to_send):
                print(f"  [{idx}] track_id={d.get('fall_track_id')}, is_fall={d.get('is_fall')}, box={d.get('box')}")

        sent_this_frame = set()  # Track what we've sent in this frame to avoid duplicates
        for det in detections:
            if det.get('send_fall_alert', False):
                track_id = det.get('fall_track_id')
                print(f"\n[TELEGRAM SEND ATTEMPT] track_id={track_id}")
                print(f"  - sent_this_frame: {sent_this_frame}")

                # Skip if already sent in this frame
                if track_id in sent_this_frame:
                    print(f"[TELEGRAM SEND SKIPPED] ⚠️ track_id={track_id} already sent in this frame")
                    continue

                # CRITICAL: Double-check cooldown before actually sending using global alert manager
                # (annotate_frame might be called multiple times)
                if not self.alert_manager.can_send_fall_alert(self.video_id, track_id):
                    print(f"[TELEGRAM SEND SKIPPED] ⚠️ Global cooldown check failed for video_id={self.video_id}, track_id={track_id}")
                    continue

                if track_id is not None and self.telegram_notifier is not None and self.telegram_notifier.enabled:
                    try:
                        # Prepare screenshot data
                        import os
                        from datetime import datetime
                        from threading import Thread
                        os.makedirs("screenshots", exist_ok=True)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        screenshot_path = f"screenshots/fall_alert_id{track_id}_{timestamp}.jpg"

                        # Clone frame to avoid race conditions
                        frame_copy = frame.copy()

                        # Send alert
                        fall_duration = det.get('dwell_time', 0)
                        centroid = det.get('centroid', (0, 0))
                        alert_message = (
                            f"🚨 FALL DETECTED 🚨\n\n"
                            f"📍 Location: {self.location_name}\n"
                            f"👤 Person ID: {track_id}\n"
                            f"⏱️ Duration: {fall_duration:.1f}s\n"
                            f"📌 Position: ({int(centroid[0])}, {int(centroid[1])})\n"
                            f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                            f"⚠️ Person ID {track_id} has been down for {fall_duration:.1f} seconds!"
                        )

                        # Save screenshot and send in background thread
                        def _save_and_send():
                            try:
                                cv2.imwrite(screenshot_path, frame_copy)
                                self.telegram_notifier.send_alert(alert_message, screenshot_path)
                            except Exception as e:
                                print(f"[TELEGRAM ERROR] ❌ Background save/send failed: {e}")

                        Thread(target=_save_and_send, daemon=True).start()
                        print(f"[TELEGRAM SENDING] 📤 Sending to Telegram (async)...")

                        # Update global cooldown timestamp AFTER successful send
                        self.alert_manager.mark_fall_alert_sent(self.video_id, track_id)
                        sent_this_frame.add(track_id)  # Mark as sent in this frame
                        print(f"[TELEGRAM SENT] ✅ Fall alert sent for video_id={self.video_id}, track_id={track_id} at {timestamp}")
                    except Exception as e:
                        print(f"[TELEGRAM ERROR] ❌ Failed to send fall alert to Telegram: {e}")

        # Check for loitering alerts to send (after fall alerts)
        loitering_alerts_to_send = [d for d in detections if d.get('send_loitering_alert', False)]
        if loitering_alerts_to_send:
            print(f"\n[TELEGRAM DEBUG] Found {len(loitering_alerts_to_send)} loitering detections marked for alert")
            for idx, d in enumerate(loitering_alerts_to_send):
                print(f"  [{idx}] track_id={d.get('id')}, loitering={d.get('loitering')}, dwell_time={d.get('dwell_time')}")

        sent_loitering_this_frame = set()  # Track what we've sent in this frame to avoid duplicates
        for det in detections:
            if det.get('send_loitering_alert', False):
                track_id = det.get('id')
                print(f"\n[TELEGRAM LOITERING SEND ATTEMPT] track_id={track_id}")
                print(f"  - sent_loitering_this_frame: {sent_loitering_this_frame}")

                # Skip if already sent in this frame
                if track_id in sent_loitering_this_frame:
                    print(f"[TELEGRAM SEND SKIPPED] ⚠️ track_id={track_id} already sent in this frame")
                    continue

                # CRITICAL: Double-check cooldown before actually sending using global alert manager
                if not self.alert_manager.can_send_loitering_alert(self.video_id, track_id):
                    print(f"[TELEGRAM SEND SKIPPED] ⚠️ Global cooldown check failed for video_id={self.video_id}, track_id={track_id}")
                    continue

                if track_id is not None and self.telegram_notifier is not None and self.telegram_notifier.enabled:
                    try:
                        # Prepare screenshot data
                        import os
                        from datetime import datetime
                        from threading import Thread
                        os.makedirs("screenshots", exist_ok=True)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        screenshot_path = f"screenshots/loitering_alert_id{track_id}_{timestamp}.jpg"

                        # Clone frame to avoid race conditions
                        frame_copy = frame.copy()

                        # Send alert
                        dwell_time = det.get('dwell_time', 0)
                        centroid = det.get('centroid', (0, 0))
                        alert_message = (
                            f"⏱️ LOITERING DETECTED ⏱️\n\n"
                            f"📍 Location: {self.location_name}\n"
                            f"👤 Person ID: {track_id}\n"
                            f"⏱️ Dwell Time: {dwell_time:.1f}s\n"
                            f"📌 Position: ({int(centroid[0])}, {int(centroid[1])})\n"
                            f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                            f"⚠️ Person ID {track_id} has been loitering for {dwell_time:.1f} seconds!"
                        )

                        # Save screenshot and send in background thread
                        def _save_and_send():
                            try:
                                cv2.imwrite(screenshot_path, frame_copy)
                                self.telegram_notifier.send_alert(alert_message, screenshot_path)
                            except Exception as e:
                                print(f"[TELEGRAM ERROR] ❌ Background save/send failed: {e}")

                        Thread(target=_save_and_send, daemon=True).start()
                        print(f"[TELEGRAM SENDING] 📤 Sending loitering alert to Telegram (async)...")

                        # Update global cooldown timestamp AFTER successful send
                        self.alert_manager.mark_loitering_alert_sent(self.video_id, track_id)
                        sent_loitering_this_frame.add(track_id)  # Mark as sent in this frame
                        print(f"[TELEGRAM SENT] ✅ Loitering alert sent for video_id={self.video_id}, track_id={track_id} at {timestamp}")
                    except Exception as e:
                        print(f"[TELEGRAM ERROR] ❌ Failed to send loitering alert to Telegram: {e}")

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
        Note: Fall and loitering alert cooldown timers are preserved to prevent duplicate alerts across video loops.
        Use reset_all() to completely reset everything including cooldown timers.
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

        # Reset fall detections (but KEEP cooldown timers to prevent duplicate alerts across video loops)
        self.fall_detections.clear()
        # DO NOT clear fall_alerts_sent or loitering_alerts_sent - we want to maintain cooldown even across video loops
        # self.fall_alerts_sent.clear()  # Commented out to prevent duplicate alerts
        # self.loitering_alerts_sent.clear()  # Commented out to prevent duplicate alerts

        # CRITICAL FIX: Reset YOLO tracker to restart IDs from 1
        # This prevents infinite ID growth when video loops
        if self.model is not None and hasattr(self.model, 'predictor') and self.model.predictor is not None:
            # Reset tracker state - forces ByteTrack to restart ID counter
            if hasattr(self.model.predictor, 'trackers'):
                trackers = self.model.predictor.trackers
                # Handle both list and dict types (newer YOLO versions use list)
                tracker_list = []
                if isinstance(trackers, dict):
                    tracker_list = list(trackers.values())
                elif isinstance(trackers, list):
                    tracker_list = trackers
                
                for tracker in tracker_list:
                    # Reset ByteTrack internal state
                    if hasattr(tracker, 'reset'):
                        tracker.reset()
                    else:
                        # Manually reset tracked objects and ID counter
                        if hasattr(tracker, 'tracked_stracks'):
                            tracker.tracked_stracks = []
                        if hasattr(tracker, 'lost_stracks'):
                            tracker.lost_stracks = []
                        if hasattr(tracker, 'removed_stracks'):
                            tracker.removed_stracks = []
                        if hasattr(tracker, 'frame_id'):
                            tracker.frame_id = 0
                        if hasattr(tracker, 'track_id_count'):
                            tracker.track_id_count = 0
                if tracker_list:
                    print(f"[DETECTOR] 🔄 ByteTrack tracker reset - IDs will restart from 1")

        # Reset color cache
        self.color_cache.clear()
        self.color_confirmed.clear()
        self.pose_cache.clear()  # Clear pose keypoints cache
        self.face_cache.clear()
        self.face_confirmed.clear()
        self.mask_cache.clear()
        self.mask_confirmed.clear()
        self.handbag_cache.clear()  # Clear handbag detection cache
        self.handbag_confirmed.clear()  # Clear handbag confirmation status
        self.track_history.clear()  # Clear trajectory lines on video loop
        
        # FIX: Clear ReID features to prevent memory leak on loop
        if hasattr(self, 'reid_features'):
            self.reid_features.clear()

        print(f"[DETECTOR] 🔄 Analytics reset for video loop - video_id={self.video_id}")

    def reset_all(self):
        """
        Completely reset ALL state including fall and loitering alert cooldown timers.
        Use this when switching to a completely new video or starting fresh.
        """
        self.reset_analytics()
        # Also clear fall and loitering alert cooldown timers
        self.fall_alerts_sent.clear()
        self.loitering_alerts_sent.clear()
        print("[RESET] All state including fall and loitering alert cooldowns have been cleared")

    def __del__(self):
        """
        Destructor - Release model references when detector is destroyed.
        This allows the model pool to free GPU memory when no detectors are using a model.
        """
        try:
            # Release model references back to the pool
            if hasattr(self, 'model_pool') and self.model_pool is not None:
                if hasattr(self, 'model') and self.model is not None:
                    self.model_pool.release_main_model()
                    print("[DETECTOR] 🔓 Released main model reference")

                if hasattr(self, 'fall_model') and self.fall_model is not None:
                    self.model_pool.release_fall_model()
                    print("[DETECTOR] 🔓 Released fall model reference")

                if hasattr(self, 'mask_model') and self.mask_model is not None:
                    self.model_pool.release_mask_model()
                    print("[DETECTOR] 🔓 Released mask model reference")

                if hasattr(self, 'pose_model') and self.pose_model is not None:
                    self.model_pool.release_pose_model()
                    print("[DETECTOR] 🔓 Released pose model reference")

                # Optional: Cleanup unused models from pool
                # self.model_pool.cleanup_unused()
        except Exception as e:
            # Silent failure in destructor to avoid issues during cleanup
            pass
