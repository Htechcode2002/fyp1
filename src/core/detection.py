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
        self.cm = ConfigManager()
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
        self.danger_threshold = danger_threshold  # Danger threshold for crowd warning
        self.danger_warning_active = False  # Track if danger warning is currently active
        self.telegram_notifier = create_telegram_notifier()  # Telegram alert system

        # OPTIMIZATION: Use shared model pool instead of individual model instances
        self.model_pool = get_model_pool()
        self.model = None
        self.display_mode = "dot"  # Default back to cleaner 'dot' mode
        self.trajectory_enabled = True  # Show trajectory lines (can be disabled for performance)
        self.show_debug_boxes = False  # DISABLED: Reducing lag and visual clutter
        self.is_recording = False  # Visual indicator state

        # Optimization: Frame Skipping & Frequency
        self.frame_count = 0
        self._last_known_count = 0 # To detect when sources are removed
        self.inference_freq = 1 # Process every frame for smooth tracking (set to 2 for better performance)
        self.heatmap_update_freq = 5 # Update heatmap overlay every 5 frames
        self.cached_heatmap_overlay = None
        self.face_analysis_freq = 4 # Default 4 for smooth performance on 1660S
        self.last_face_info = []    # Persistent results for smooth drawing
        self.face_ai_sleep_until = 0 # If efficiency is low, sleep for N frames

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
        self.face_confirmed = {} # track_id -> bool

        # Performance Monitoring
        self._perf_stats = {
            "inference": 0.0,
            "latency": 0.0,
            "fps": 0.0,
            "load": 0.0
        }
        self.face_confirmed = {} # track_id -> bool (True if confirmed - stop detecting)
        self.face_attempts = {} # track_id -> int (number of failed attempts)
        self.face_samples = {} # track_id -> list of samples (legacy, used for "Calculating..." display)

        # Mask Detection Caching (1-confirmation) - OPTIMIZED
        self.mask_cache = {} # track_id -> mask_status_string (final confirmed)
        self.mask_confirmed = {} # track_id -> bool (True if confirmed - stop detecting)
        self.mask_samples = {} # track_id -> list of samples (legacy, for cleanup)

        # Handbag Detection Tracking (1-confirmation caching)
        self.handbag_cache = {} # track_id -> 1 if has handbag, 0 if no handbag (final confirmed)
        self.handbag_confirmed = {} # track_id -> bool (True if confirmed - stop detecting)

        # Backpack Detection Tracking (1-confirmation caching)
        self.backpack_cache = {} # track_id -> 1 if has backpack, 0 if no backpack (final confirmed)
        self.backpack_confirmed = {} # track_id -> bool (True if confirmed - stop detecting)

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

    def _get_optimized_path(self, base_path):
        """Helper to prefer .engine over .pt if it exists"""
        engine_path = base_path.replace(".pt", ".engine")
        if os.path.exists(engine_path):
            return engine_path
        return base_path

    def _setup_model_device(self, model, model_path):
        """Safely move model to GPU if it's a PyTorch model"""
        import torch
        if model is None: return
        
        if torch.cuda.is_available():
            device = 'cuda:0'
            # Check if it's an exported format (TensorRT, ONNX)
            is_exported = not model_path.lower().endswith('.pt')
            
            if hasattr(model, 'device') and 'cuda' in str(model.device):
                return # Already on GPU
            
            if not is_exported:
                try:
                    model.to(device)
                    print(f"[GPU] ✅ Moved PyTorch model to GPU: {os.path.basename(model_path)}")
                except: pass
            else:
                print(f"[GPU] ⚙️ Using Native {model_path.split('.')[-1].upper()} on CUDA: {os.path.basename(model_path)}")

    def load_face_model(self):
        """Load YOLO face detection model from shared pool."""
        try:
            from src.core.config_manager import ConfigManager
            cm = ConfigManager()
            face_model_path = cm.get("yolo", {}).get("face_model_path", "models/face/yolov8n-face.pt")
            face_model_path = face_model_path.replace("\\", "/")
            
            # AUTOMATIC OPTIMIZATION: Prefer .engine
            face_model_path = self._get_optimized_path(face_model_path)
            
            print(f"[DETECTOR] 🔗 Getting shared YOLO Face model from pool (path: {face_model_path})...")
            face_model = self.model_pool.get_face_model(face_model_path)
            if face_model is not None:
                self.face_analyzer.set_yolo_face_model(face_model)
                self._setup_model_device(face_model, face_model_path)
        except Exception as e:
            print(f"[DETECTOR] ❌ ERROR: Failed to load face detection model: {e}")

    def load_fall_detection_model(self):
        """Load fall detection model from shared pool (GPU optimized)"""
        base_path = "models/fall/fall_det_1.pt"
        fall_model_path = self._get_optimized_path(base_path)

        if not os.path.exists(fall_model_path) and not os.path.exists(base_path):
            print(f"[DETECTOR] ⚠️ Fall model not found. Skipping.")
            return

        try:
            self.fall_model = self.model_pool.get_fall_model(fall_model_path)
            self._setup_model_device(self.fall_model, fall_model_path)
        except Exception as e:
            print(f"[DETECTOR] ❌ ERROR: Failed to load fall detection: {e}")
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

        # White - high brightness, low saturation (OPTIMIZED for better accuracy)
        # Use gradient thresholds: brighter = more tolerant to saturation
        if v > 200 and s < 60:  # Very bright → definitely white
            return "White"
        if v > 160 and s < 40:  # Bright → white if very low saturation
            return "White"
        if v > 140 and s < 25:  # Medium-bright → white if extremely low saturation
            return "White"

        # Gray - low saturation, medium brightness (OPTIMIZED: exclude white range)
        if s < 60 and 70 <= v <= 140:
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

            # SKIP independent YOLO face model to save GPU RAM (using Fast Global Mode instead)
            # if self.face_analysis_enabled:
            #     self.load_face_model()

            self._models_loaded = True
            self._models_loading = False
            print(f"[DETECTOR] ✅ All models loaded for video_id={self.video_id}")

        # Start background loading thread
        import threading
        threading.Thread(target=_load_models_async, daemon=True).start()

        return False  # Not ready yet, will be ready on next frame

    def load_model(self):
        """Load main YOLO model from shared model pool (GPU optimized)"""
        cm = ConfigManager()
        base_path = cm.get("yolo", {}).get("model_path", "models/yolov8n.pt")
        model_path = self._get_optimized_path(base_path)
        
        print(f"[DETECTOR] 📋 Configured model path: {model_path}")

        try:
            self.model = self.model_pool.get_main_model(model_path)
            self._setup_model_device(self.model, model_path)
        except Exception as e:
            print(f"[DETECTOR] ❌ CRITICAL ERROR: Failed to load YOLO: {e}")
            self.model = None


    def load_mask_model(self):
        """Load YOLO mask detection model from shared pool (GPU optimized)"""
        base_path = "models/mask/mask.pt"
        mask_model_path = self._get_optimized_path(base_path)

        try:
            self.mask_model = self.model_pool.get_mask_model(mask_model_path)
            self._setup_model_device(self.mask_model, mask_model_path)
        except Exception as e:
            print(f"[DETECTOR] ❌ Failed to load Mask model: {e}")
            self.mask_model = None

    def _get_shirt_box_simple(self, x1, y1, x2, y2, frame_shape):
        """Ultra-fast clothes ROI estimation using only the person's bounding box"""
        hf, wf = frame_shape[:2]
        bw = x2 - x1
        bh = y2 - y1
        
        # Human torso is approximately in the upper-middle region
        # ROI: 20-50% of height, 25-75% of width
        sx1 = int(x1 + bw * 0.25)
        sy1 = int(y1 + bh * 0.15)
        sx2 = int(x2 - bw * 0.25)
        sy2 = int(y1 + bh * 0.45)
        
        # Clip to boundaries
        sx1, sy1 = max(0, sx1), max(0, sy1)
        sx2, sy2 = min(wf, sx2), min(hf, sy2)
        
        return [sx1, sy1, sx2, sy2]

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
        # OPTIMIZATION: Increased from 300 to 5000. Frequent empty_cache() causes execution stutters
        # which breaks real-time tracking consistency and drops detection rates.
        if self.frame_count % 5000 == 0:  # Every ~3 minutes at 30 FPS
            try:
                import torch
                import gc
                if torch.cuda.is_available():
                    # Forcing GC and empty_cache is expensive (causes stutter)
                    # Only do it very infrequently to prevent memory fragmentation
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    allocated = torch.cuda.memory_allocated() / 1024**2
                    reserved = torch.cuda.memory_reserved() / 1024**2
                    print(f"[GPU] Maintenance: {allocated:.0f}MB used (Cache Refreshed)")
            except Exception:
                pass

        # CRITICAL: Monitor cache sizes to detect memory leaks
        if self.frame_count % 300 == 0:  # Check every 300 frames (every 10 seconds at 30 FPS)
            cache_sizes = {
                'color_cache': len(self.color_cache),
                'face_cache': len(self.face_cache),
                'mask_cache': len(self.mask_cache),
                'handbag_cache': len(self.handbag_cache),
                'backpack_cache': len(self.backpack_cache),
                'track_history': len(self.track_history),
                'seen_track_ids': len(self.seen_track_ids)
            }
            total_cache_size = sum(cache_sizes.values())
            if total_cache_size > 500:  # Warning threshold
                print(f"⚠️ WARNING: Large cache detected ({total_cache_size} entries): {cache_sizes}")
                print(f"   Consider checking if video is looping correctly and reset_analytics() is being called")

        # PERFORMANCE OPTIMIZATION: Adaptive frame skipping for multi-stream stability
        # We calculate load based on total active cameras
        source_count = self.cm.get_active_count()
        
        # MONITOR: If count changed significantly (reduced), reload config and flush VRAM
        if source_count < self._last_known_count:
            print(f"[OPTIMIZER] 📉 Source count reduced ({self._last_known_count} -> {source_count}). Flushing caches...")
            import torch
            import gc
            self.cm.load_config() # Force sync singleton
            source_count = self.cm.get_active_count() # Get fresh count
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
        self._last_known_count = source_count

        # Dynamic skip strategy:
        # 1-2 sources: Full speed (skip_freq=1)
        # 3-4 sources: Skip every other frame (skip_freq=2)
        # 5+ sources: Heavy optimization (skip_freq=3+)
        self._perf_stats["load"] = source_count # Record active streams
        skip_freq = self.inference_freq

        # CRITICAL FIX: Single-stream "High Sensitivity" mode
        if source_count <= 1:
            imgsz = 640  # Stay at 640 to avoid 'AssertionError' with fixed-size models
            conf_threshold = 0.10 # Lower threshold to catch small/distant people
        else:
            imgsz = 640
            conf_threshold = 0.15
        
        # MONITOR: Warn if FPS is critically low for tracking stability
        if self._perf_stats["fps"] < 8 and self.frame_count % 100 == 0:
            print(f"⚠️ CRITICAL PERFORMANCE WARNING: Running at {self._perf_stats['fps']:.1f} FPS.")
            print(f"   Tracker stability is highly compromised below 10 FPS. People may 'vanish'.")
            print(f"   SUGGESTION: Disable 'Face Analysis' or 'Mask Detection' to recover tracking quality.")
        
        # Check if we should skip detection but keep last known state for visuals
        if self.frame_count % skip_freq != 0 and self.last_detections:
            return self.last_detections, self.last_enhanced_counts
            
        detections = []
        import copy
        current_counts = copy.deepcopy(self.line_counts)
        current_time = time.time()
        t_detect_start = time.time()
        
        # Balanced settings handled by source_count logic above
        
        t_inf_start = time.time()
        if tracking_enabled:
            # Using higher imgsz for single stream accuracy
            results = self.model.track(frame, persist=True, verbose=False, classes=[0], tracker=self.tracker, imgsz=imgsz, conf=conf_threshold, iou=0.7, half=True)
        else:
            # Predict only (Detection) - No IDs
            results = self.model.predict(frame, verbose=False, classes=[0], imgsz=imgsz, conf=conf_threshold, iou=0.7, half=True)
        self._perf_stats["inference"] = (time.time() - t_inf_start) * 1000

        # PERFORMANCE OPTIMIZATION: Staggered Inference (分时调度)
        # Instead of running all models on one frame, we spread them across 4 frames
        # to prevent GPU load spikes that cause the "0.5s freeze".
        cycle = self.frame_count % 4
        
        # Frame 0: Reserved for Main YOLO + Tracking (Already done above)
        
        # Frame 1: Pose Estimation + Color Detection
        # Frame 1: Color Detection (Now simplified)
        pose_results = None
        
        # We need to preserve run_secondary for backward compatibility in the code flow
        # but now it's staggered within the specific blocks below

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

        # First pass: Collect unconfirmed people
        for result in results:
            boxes = result.boxes
            if tracking_enabled and boxes.id is not None:
                ids = boxes.id.cpu().numpy().astype(int)
            else:
                ids = [None] * len(boxes)
            cls_ids = boxes.cls.cpu().numpy().astype(int)

            for box, track_id, cls_id in zip(boxes, ids, cls_ids):
                if cls_id == 0:  # Person
                    # OPTIMIZATION: Only add to detection list if NOT confirmed yet
                    if track_id is not None and not self.handbag_confirmed.get(track_id, False):
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                        person_detections.append({
                            'track_id': track_id,
                            'box': (x1, y1, x2, y2),
                            'centroid': (cx, cy)
                        })

        # Second pass: Only collect handbags if we have unconfirmed people (saves processing)
        if person_detections:
            # OPTIMIZATION: Skip handbag detection if all people are already confirmed
            for result in results:
                boxes = result.boxes
                cls_ids = boxes.cls.cpu().numpy().astype(int)

                for box, cls_id in zip(boxes, cls_ids):
                    if cls_id == 26:  # Handbag (class 26 in COCO)
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                        handbag_detections.append({
                            'box': (x1, y1, x2, y2),
                            'centroid': (cx, cy)
                        })

        # Associate handbags with nearby persons (ONLY if we have unconfirmed people AND handbags)
        # Skip association if no unconfirmed people (performance optimization)
        if person_detections and handbag_detections:
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

        # === BACKPACK DETECTION: First pass to collect all detections ===
        # OPTIMIZATION: Only detect for people who DON'T have confirmed backpack status
        person_detections_bp = []  # list of {track_id, box, centroid} - only unconfirmed people
        backpack_detections = []  # list of {box, centroid}

        # First pass: Collect unconfirmed people
        for result in results:
            boxes = result.boxes
            if tracking_enabled and boxes.id is not None:
                ids = boxes.id.cpu().numpy().astype(int)
            else:
                ids = [None] * len(boxes)
            cls_ids = boxes.cls.cpu().numpy().astype(int)

            for box, track_id, cls_id in zip(boxes, ids, cls_ids):
                if cls_id == 0:  # Person
                    # OPTIMIZATION: Only add to detection list if NOT confirmed yet
                    if track_id is not None and not self.backpack_confirmed.get(track_id, False):
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                        person_detections_bp.append({
                            'track_id': track_id,
                            'box': (x1, y1, x2, y2),
                            'centroid': (cx, cy)
                        })

        # Second pass: Only collect backpacks if we have unconfirmed people (saves processing)
        if person_detections_bp:
            # OPTIMIZATION: Skip backpack detection if all people are already confirmed
            for result in results:
                boxes = result.boxes
                cls_ids = boxes.cls.cpu().numpy().astype(int)

                for box, cls_id in zip(boxes, cls_ids):
                    if cls_id == 24:  # Backpack (class 24 in COCO)
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                        backpack_detections.append({
                            'box': (x1, y1, x2, y2),
                            'centroid': (cx, cy)
                        })

        # Associate backpacks with nearby persons (ONLY if we have unconfirmed people AND backpacks)
        if person_detections_bp and backpack_detections:
            for person in person_detections_bp:
                track_id = person['track_id']
                if track_id is None:
                    continue

                px, py = person['centroid']
                p_x1, p_y1, p_x2, p_y2 = person['box']

                # Check for nearby backpacks (within reasonable distance)
                has_backpack = 0
                for backpack in backpack_detections:
                    bx, by = backpack['centroid']

                    # Calculate distance between person and backpack
                    distance = np.sqrt((px - bx)**2 + (py - by)**2)

                    # Check if backpack is near person (distance threshold)
                    person_height = p_y2 - p_y1
                    max_distance = person_height * 0.8  # Backpack within 80% of person height

                    if distance < max_distance:
                        has_backpack = 1
                        break

                # Cache the result and CONFIRM - stop future detection for this track_id
                self.backpack_cache[track_id] = has_backpack
                self.backpack_confirmed[track_id] = True  # CONFIRMED - stop detecting
                if has_backpack:
                    print(f"[BACKPACK CONFIRMED] Track ID {track_id}: Has backpack")
                else:
                    print(f"[BACKPACK CONFIRMED] Track ID {track_id}: No backpack")

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
                keypoints = None # CRITICAL FIX: Initialize here so it exists for ALL classes

                # Only track People (Class 0) for counting logic to avoid counting bags as people
                if cls_id == 0:
                    # Centroid
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    current_point = (cx, cy)

                    # --- COLOR DETECTION (Simple ROI-based) ---
                    # Handle Color Detection with "Best Sample" logic
                    if track_id is not None and self.color_confirmed.get(track_id, False):
                        shirt_color = self.color_cache.get(track_id, "Unknown")
                        shirt_box = self._get_shirt_box_simple(x1, y1, x2, y2, frame.shape)
                    else:
                        shirt_box = self._get_shirt_box_simple(x1, y1, x2, y2, frame.shape)
                        if shirt_box is not None:
                            sx1, sy1, sx2, sy2 = shirt_box
                            if sx2 > sx1 and sy2 > sy1:
                                shirt_roi = frame[sy1:sy2, sx1:sx2]
                                if shirt_roi.size > 0:
                                    detected_color = self.get_dominant_color(shirt_roi)
                                    
                                    # QUALITY CHECK: Only confirm if color is distinct OR we've tried many times
                                    attempts = self.face_attempts.get(f"col_{track_id}", 0)
                                    self.face_attempts[f"col_{track_id}"] = attempts + 1
                                    
                                    # If we got a real color (not Unknown), or after 15 attempts, we lock it in
                                    is_good_sample = detected_color not in ["Unknown"]
                                    
                                    if track_id is not None:
                                        if is_good_sample or attempts > 15:
                                            self.color_cache[track_id] = detected_color
                                            self.color_confirmed[track_id] = True
                                            print(f"[COLOR CONFIRMED] Track ID {track_id}: {detected_color} (at attempt {attempts})")
                                            shirt_color = detected_color
                                        else:
                                            # Still trying to get a better color
                                            shirt_color = detected_color

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
                    
                        # --- PERFORMANCE OPTIMIZATION: Distance-Based Inference ---
                        # Only run heavy sub-models if the person is large enough to see details
                        person_area = (x2 - x1) * (y2 - y1)
                        is_detailed_enough = person_area > (frame.shape[0] * frame.shape[1] * 0.005) # ~0.5% of total image
                        
                        # --- FALL DETECTION (POSE BASED) ---
                        is_fall = False
                        if self.fall_detection_enabled and is_detailed_enough:
                            # Previous logic for fall...
                            pass
                        
                        # --- BAG DETECTION (IF DETAILED) ---
                        if is_detailed_enough:
                            # Original bag detection logic here...
                            pass        
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

                                        # Get Backpack Status from cache if available
                                        backpack_val = 0
                                        if track_id is not None and track_id in self.backpack_cache:
                                            backpack_val = self.backpack_cache[track_id]

                                        print(f"[DEBUG] Line crossing - track_id: {track_id}, mask_val: {mask_val}, handbag: {handbag_val}, backpack: {backpack_val}")

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
                                            handbag=handbag_val,
                                            backpack=backpack_val
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
                        # --- TRAJECTORY JUMP PROTECTION ---
                        # If the point jumps too far in one frame, it's probably an ID switch due to lag.
                        # We clear the history to prevent long diagonal lines.
                        if track_id in self.track_history and self.track_history[track_id]:
                            last_x, last_y = self.track_history[track_id][-1]
                            dist = np.sqrt((cx - last_x)**2 + (cy - last_y)**2)
                            # Threshold: 30% of frame width is a reasonable "impossible" jump for 1 frame
                            if dist > frame.shape[1] * 0.3:
                                self.track_history[track_id] = []

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
                if tid in self.face_attempts:
                    del self.face_attempts[tid]
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

        # Frame 2: Fall Detection (Pose Based)
        # PERFORMANCE OPTIMIZATION: Staggered on cycle == 2
        if self.fall_detection_enabled and self.fall_model is not None and cycle == 2:
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
                    # OPTIMIZATION: Use half precision (FP16) and smaller imgsz for faster inference
                    # TensorRT requires exact size (640)
                    fall_results = self.fall_model(frame, verbose=False, half=True, imgsz=640)
                    current_fall_boxes = []

                    # Collect all detected falls in this frame and match to track_id
                    for fall_result in fall_results:
                        fall_boxes = fall_result.boxes
                        if fall_boxes is not None and len(fall_boxes) > 0:
                            for fall_box in fall_boxes:
                                fx1, fy1, fx2, fy2 = fall_box.xyxy[0].cpu().numpy()
                                fall_conf = float(fall_box.conf[0].cpu().numpy())
                                
                                # STAGE 1: Confidence Check (Strict for falls)
                                if fall_conf < 0.6: 
                                    continue
                                
                                # STAGE 2: Aspect Ratio Validation
                                # A real fall results in a horizontal or roughly square bounding box.
                                fw = fx2 - fx1
                                fh = fy2 - fy1
                                aspect_ratio = fw / fh if fh > 0 else 0
                                
                                # If the person is still more vertical than horizontal, it's likely 
                                # just a squat or a bend, not a prismatic fall.
                                if aspect_ratio < 0.85:
                                    continue
                                
                                box = [int(fx1), int(fy1), int(fx2), int(fy2)]

                                # Find the person this fall belongs to (highest IoU)
                                best_match_track_id = None
                                best_iou = 0.0
                                for person_data in person_boxes_with_ids:
                                    iou = self._calculate_iou(box, person_data['box'])
                                    if iou > best_iou:
                                        best_iou = iou
                                        best_match_track_id = person_data['track_id']

                                # STAGE 4: IoU Overlap (Ensure it's actually a person falling)
                                if best_iou > 0.45:
                                    current_fall_boxes.append({
                                        'box': box,
                                        'conf': fall_conf,
                                        'centroid': (int((fx1+fx2)/2), int((fy1+fy2)/2)),
                                        'track_id': best_match_track_id
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

        # --- FACE ANALYSIS: Global Fast Mode (Dynamic Load Scaling) ---
        if self.face_analysis_enabled and self.face_analyzer.enabled:
            # DYNAMIC LOAD SCALING: Scale AI frequency based on crowd density
            person_count = len([d for d in detections if d['cls_id'] == 0])
            if person_count > 10:
                self.face_analysis_freq = 10 # High load, slow down AI
            elif person_count > 5:
                self.face_analysis_freq = 6 # Medium load
            else:
                self.face_analysis_freq = 4 # Minimum 4 frames to prevent GPU stuttering
            
            # Frame 3: Face Analysis (Staggered on cycle == 3 AND obeys dynamic freq)
            if cycle == 3 and (self.frame_count // 4) % (self.face_analysis_freq // 4 + 1) == 0:
                try:
                    # ROI EXTRACRTION: Find people who need face analysis
                    unconfirmed_with_areas = []
                    for det in detections:
                        if det['cls_id'] == 0:
                            bw = det['box'][2] - det['box'][0]
                            bh = det['box'][3] - det['box'][1]
                            # LOOSENED: Allow smaller crops for distant faces while prioritizing close ones
                            if bw > 20 and bh > 40:
                                area = bw * bh
                                unconfirmed_with_areas.append((area, det['box']))

                    # PRIORITIZE: Sort by area (descending) so closest people are analyzed first
                    unconfirmed_with_areas.sort(key=lambda x: x[0], reverse=True)
                    roi_boxes = [r[1] for r in unconfirmed_with_areas]

                    # Run ROI-based face detection (Ultra Fast & Accurate)
                    face_info = self.face_analyzer.analyze_faces_global(frame, roi_boxes=roi_boxes)
                    self.last_face_info = face_info
                    
                    # 2. MATCH BACK TO PEOPLE (To show in main label)
                    for face in face_info:
                        f_center = ((face['bbox'][0]+face['bbox'][2])/2, (face['bbox'][1]+face['bbox'][3])/2)
                        
                        # Find closest person detection
                        closest_det = None
                        min_dist = float('inf')
                        for det in detections:
                            if det['cls_id'] == 0:
                                p_center = ((det['box'][0]+det['box'][2])/2, (det['box'][1]+det['box'][3])/2)
                                dist = np.sqrt((f_center[0]-p_center[0])**2 + (f_center[1]-p_center[1])**2)
                                if dist < min_dist:
                                    min_dist = dist
                                    closest_det = det
                        
                        # Lock result if valid distance (< 75% height)
                        if closest_det and min_dist < (closest_det['box'][3]-closest_det['box'][1]) * 0.75:
                            tid = closest_det.get('id')
                            if tid is not None:
                                # PREMIUM: Age & Gender Smoothing (Median Filtering)
                                if tid not in self.face_cache:
                                    self.face_cache[tid] = {'genders': [], 'ages': []}
                                
                                self.face_cache[tid]['genders'].append(face['gender'])
                                self.face_cache[tid]['ages'].append(face['age'])
                                
                                # Keep last 15 samples for stability
                                if len(self.face_cache[tid]['ages']) > 15:
                                    self.face_cache[tid]['genders'].pop(0)
                                    self.face_cache[tid]['ages'].pop(0)
                                
                                self.face_attempts[tid] = self.face_attempts.get(tid, 0) + 1
                                if self.face_attempts[tid] >= 3:
                                    self.face_confirmed[tid] = True
                except Exception as e:
                    print(f"[FACE] Analysis Error: {e}")
            
            # --- 3. PERSISTENT UPDATER: Apply results to detections on EVERY frame ---
            for det in detections:
                if det['cls_id'] == 0:
                    tid = det.get('id')
                    if tid is not None and tid in self.face_cache and self.face_cache[tid]['ages']:
                        # Calculate stable results from cache
                        stable_age = int(np.median(self.face_cache[tid]['ages']))
                        males = self.face_cache[tid]['genders'].count('Male')
                        stable_gender = 'Male' if males >= len(self.face_cache[tid]['genders'])/2 else 'Female'
                        
                        det['gender'] = stable_gender
                        det['age'] = stable_age
                        # Cache stable values for DB logging retrieval
                        self.face_cache[tid]['gender'] = stable_gender
                        self.face_cache[tid]['age'] = stable_age

        # --- MASK DETECTION: Persistent Results (Prevents Flickering) ---
        if self.mask_detection_enabled:
            for det in detections:
                if det['cls_id'] == 0:
                    tid = det.get('id')
                    if tid is not None and tid in self.mask_cache:
                        mask_str = self.mask_cache[tid]
                        det['mask_status_str'] = mask_str
                        mask_labels_reverse = {"With Mask": 0, "No Mask": 1, "Mask Incorrect": 2}
                        det['mask_status'] = mask_labels_reverse.get(mask_str, None)

        # --- MASK DETECTION ENGINE (Runs only on cycle 1) ---
        # PERFORMANCE: Only run mask detection on Frame 1 to stagger load
        if self.mask_detection_enabled and self.mask_model is not None and cycle == 1:
            # print(f"[DEBUG] Mask detection ENABLED and running on frame")
            try:
                # Skip the redundant loop here as we handle persistence above

                # Run mask detection ONLY for unconfirmed people
                people_to_detect = [det for det in detections if det['cls_id'] == 0 and not self.mask_confirmed.get(det.get('id'), False)]

                if people_to_detect:
                    # OPTIMIZATION: Use smaller imgsz (480) for mask detection
                    # TensorRT requires exact size (640)
                    mask_results = self.mask_model(frame, verbose=False, half=True, imgsz=640)

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
        
        # Add performance stats to output
        self._perf_stats["latency"] = (time.time() - t_detect_start) * 1000
        self._perf_stats["fps"] = 1000.0 / self._perf_stats["latency"] if self._perf_stats["latency"] > 0 else 0
        
        # MONITOR: Warn if FPS is critically low for tracking stability
        if self._perf_stats["fps"] < 8 and self.frame_count % 100 == 0:
            print(f"⚠️ CRITICAL PERFORMANCE WARNING: Running at {self._perf_stats['fps']:.1f} FPS.")
            print(f"   Tracker stability is highly compromised below 10 FPS. People may 'vanish'.")
            print(f"   SUGGESTION: Disable 'Face Analysis' or 'Mask Detection' to recover tracking quality.")

        enhanced_counts["_perf"] = self._perf_stats
        
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

            # OPTIMIZATION: Skip drawing backpack/handbag objects - we only show [BP]/[BAG] labels on people
            if cls_id == 24 or cls_id == 26:
                continue  # Skip bag objects, only show person attributes

            # Default Colors (BGR)
            color = (0, 255, 0) # Green for Person
            label_prefix = "ID"

            if is_fall:
                color = (0, 0, 255)
                label_prefix = "FALL"

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

            # Draw Shirt Sampling Box (only during debugging)
            if self.show_debug_boxes and det.get("shirt_box") and track_id is not None:
                # Check if color is confirmed
                is_confirmed = self.color_confirmed.get(track_id, False)

                # ALWAYS show box for user visualization
                if True:
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

            # (Skeleton visualization removed for performance)

            # Draw Mask Bounding Box (only if NOT confirmed - during confirmation phase)
            # OPTIMIZATION: Only show debug boxes if enabled
            if self.show_debug_boxes and det.get("mask_box") and track_id is not None:
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

            if is_fall:
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

                 # Append Gender and Age if available (REAL-TIME DISPLAY)
                 if "gender" in det and "age" in det:
                     if det["gender"] is not None and det["age"] is not None:
                         gender_short = "M" if det["gender"] == "Male" else "F"
                         label += f" {gender_short}/{det['age']}"
                 
                 # (Space preserved)

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

                 # Append Backpack Status if available
                 if track_id is not None and self.backpack_confirmed.get(track_id, False):
                     has_backpack = self.backpack_cache.get(track_id, 0)
                     if has_backpack == 1:
                         label += " [BP]"  # Has backpack

            if label:
                # Draw Label with background
                label_text_color = (255, 255, 255)
                t_size = cv2.getTextSize(label, 0, fontScale=0.6, thickness=2)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 5
                
                # Background Rect
                cv2.rectangle(frame, (x1, y1), c2, label_bg_color, -1, cv2.LINE_AA)
                
                # Text
                cv2.putText(frame, label, (x1, y1 - 2), 0, 0.6, label_text_color, thickness=2, lineType=cv2.LINE_AA)

            if self.trajectory_enabled and track_id is not None and track_id in self.track_history:
                points = self.track_history[track_id]
                if len(points) > 1:
                    # Move Average Smoothing for Trajectories
                    smoothed_pts = []
                    window = 3
                    for k in range(len(points)):
                        start = max(0, k - window)
                        end = k + 1
                        avg_pt = np.mean(points[start:end], axis=0).astype(int)
                        smoothed_pts.append(avg_pt)
                    
                    pts = np.array(smoothed_pts, np.int32).reshape((-1, 1, 2))
                    # Draw with gradient or thick line
                    cv2.polylines(frame, [pts], False, (0, 255, 255), 2, cv2.LINE_AA)
            
        # --- FACE Diagnostic Brackets Removed ---
        # if detections and self.face_analysis_enabled and hasattr(self, 'last_face_info') and self.last_face_info:
        #     self.face_analyzer.draw_face_info(frame, self.last_face_info)
            
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

        # --- RECORDING INDICATOR (Top Right) ---
        if self.is_recording:
            h, w = frame.shape[:2]
            # Calculate pulsing effect based on time (2Hz pulse)
            pulse = (int(time.time() * 2) % 2) == 0
            
            # Position and Size
            margin_right = 30
            margin_top = 30
            pill_h = 50
            pill_w = 160
            
            # Draw semi-transparent background "pill" for maximum visibility
            overlay = frame.copy()
            cv2.rectangle(overlay, (w - pill_w - margin_right, margin_top), (w - margin_right, margin_top + pill_h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
            
            dot_color = (0, 0, 255) if pulse else (0, 0, 100) # Bright red vs Dull red
            dot_radius = 12
            dot_center = (w - pill_w - margin_right + 30, margin_top + pill_h // 2)
            
            # Draw the pulsing red dot
            cv2.circle(frame, dot_center, dot_radius, dot_color, -1, cv2.LINE_AA)
            if pulse: # Extra glow when bright
                 cv2.circle(frame, dot_center, dot_radius + 4, (0, 0, 255), 2, cv2.LINE_AA)
            
            # Draw large bold "REC" text
            cv2.putText(frame, "REC", (w - pill_w - margin_right + 60, margin_top + 38), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)

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

    def set_recording_state(self, is_recording):
        """Toggle the visual 'REC' indicator on the video frame"""
        self.is_recording = is_recording

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
        self.face_cache.clear()
        self.face_confirmed.clear()
        self.mask_cache.clear()
        self.mask_confirmed.clear()
        self.handbag_cache.clear()  # Clear handbag detection cache
        self.handbag_confirmed.clear()  # Clear handbag confirmation status
        self.backpack_cache.clear()  # Clear backpack detection cache
        self.backpack_confirmed.clear()  # Clear backpack confirmation status
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
        """
        try:
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
                # (Pose model release removed)
                    print("[DETECTOR] 🔓 Released pose model reference")
                if hasattr(self, 'face_analyzer'):
                    # The analyzer logic is shared, we just release the count
                    self.model_pool.release_face_model()
                    print("[DETECTOR] 🔓 Released face model reference")
            
            # Explicitly clear CUDA cache to prevent memory buildup
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("[GPU] 🧹 Forced VRAM cleanup completed")
        except Exception:
            pass
