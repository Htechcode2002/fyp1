from ultralytics import YOLO
import cv2
import numpy as np
from src.core.config_manager import ConfigManager
from src.core.database import DatabaseManager
import os
import time
from datetime import datetime

class VideoDetector:
    def __init__(self, tracker="bytetrack.yaml", location_name=None, video_id=None):
        self.tracker = tracker
        self.location_name = location_name if location_name else "Unknown Location"
        self.video_id = video_id
        self.db = DatabaseManager()
        self.heatmap_enabled = False
        self.heatmap_accumulator = None
        self.model = None
        self.load_model()

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

        # Line crossing flash effect
        self.line_flash = {} # line_idx -> timestamp of last crossing

    def set_heatmap(self, enabled):
        self.heatmap_enabled = enabled

    def get_dominant_color(self, roi):
        """
        High-fidelity nearest-neighbor color classifier.
        roi: BGR image crop (numpy array)
        """
        if roi.size == 0:
            return "Unknown"
            
        # 1. Average in BGR
        mean_bgr = np.mean(roi, axis=(0, 1))
        
        # 2. Convert reference point to HSV
        mean_pixel = np.array([[mean_bgr]], dtype=np.uint8)
        hsv_pixel = cv2.cvtColor(mean_pixel, cv2.COLOR_BGR2HSV)
        h, s, v = hsv_pixel[0][0]
        
        # --- USER PROVIDED SIMPLE LOGIC ---
        # OpenCV HSV ranges: H(0-179), S(0-255), V(0-255)
        
        # Dark colors (Black) often have some hue noise but low Value
        # Refined: Only call it Black if it's REALLY dark, OR if it's dark and unsaturated.
        # This allows "Navy Blue" (Dark but Saturated) to be detected as Blue.
        if v < 25: return "Black"
        if v < 85 and s < 90: return "Black"
        
        # Achromatic (White/Gray) - Low Saturation
        if s < 40:
            if v > 190: return "White"
            return "Gray"
            
        # Chromatic colors
        if h < 10 or h > 160: return "Red"
        if h < 25: return "Orange"
        if h < 35: return "Yellow"
        if h < 95: return "Green" # Expanded to include Cyan/Teal
        if h < 135: return "Blue" # Expanded Blue
        return "Purple"

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

        # Results holder
        results = []
        
        if tracking_enabled:
            # Persist=True for tracking
            # Classes: 0=Person, 24=Backpack, 26=Handbag
            results = self.model.track(frame, persist=True, verbose=False, classes=[0, 24, 26], tracker=self.tracker)
        else:
            # Predict only (Detection) - No IDs
            results = self.model.predict(frame, verbose=False, classes=[0, 24, 26])

        detections = []
        # Return deep copy of counts to avoid threading issues
        import copy
        current_counts = copy.deepcopy(self.line_counts)
        current_time = time.time()

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
                
                # Only track People (Class 0) for counting logic to avoid counting bags as people
                if cls_id == 0 and track_id is not None:
                    # Track unique pedestrians
                    current_frame_ids.add(track_id)
                    self.seen_track_ids.add(track_id)

                    # Centroid
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    current_point = (cx, cy)
                    
                    # --- COLOR DETECTION (Moved Up for logging) ---
                    shirt_color = "Unknown"
                    shirt_box = None
                    if cls_id == 0: # Person
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

                        # --- COLOR DETECTION (Moved Up for logging) ---
                        w_box = x2 - x1
                        h_box = y2 - y1
                        
                        # Take central 20% of width and 10% of height around center
                        half_w = (w_box * 0.2) / 2
                        half_h = (h_box * 0.1) / 2
                        
                        sx1 = int(cx - half_w)
                        sy1 = int(cy - half_h)
                        sx2 = int(cx + half_w)
                        sy2 = int(cy + half_h)
                        
                        # Clip to frame/box
                        sx1 = max(int(x1), min(sx1, int(x2)))
                        sx2 = max(int(x1), min(sx2, int(x2)))
                        sy1 = max(int(y1), min(sy1, int(y2)))
                        sy2 = max(int(y1), min(sy2, int(y2)))
                        
                        if sx2 > sx1 and sy2 > sy1:
                            shirt_roi = frame[sy1:sy2, sx1:sx2]
                            if shirt_roi.size > 0:
                                shirt_color = self.get_dominant_color(shirt_roi)
                                shirt_box = [sx1, sy1, sx2, sy2]
                    
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

                detections.append({
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                    "conf": conf,
                    "id": int(track_id) if track_id is not None else None,
                    "cls_id": int(cls_id), 
                    "loitering": is_loitering,
                    "dwell_time": dwell_time,
                    "direction_arrow": direction_arrow,
                    "centroid": (int((x1+x2)/2), int((y1+y2)/2)),
                    "shirt_color": shirt_color,
                    "shirt_box": shirt_box
                })

        # Update active track IDs
        self.active_track_ids = current_frame_ids

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
            conf = det["conf"]
            
            # Green box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Label
            # Get Attributes
            track_id = det.get("id") # Returns None if not present or explicit None
            cls_id = det.get("cls_id", 0)
            is_loitering = det.get("loitering", False)
            dwell_time = det.get("dwell_time", 0)
            
            # Draw Shirt Sampling Box (if available)
            if det.get("shirt_box"):
                sx1, sy1, sx2, sy2 = det["shirt_box"]
                # Draw small yellow rectangle to show sampling area
                cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (0, 255, 255), 1)

            # Default Colors (BGR)
            # Person (0): Green
            color = (0, 255, 0)
            label_prefix = "ID"
            
            if cls_id == 24: # Backpack
                color = (255, 0, 0) # Blue
                label_prefix = "BP" # Backpack
            elif cls_id == 26: # Handbag
                color = (255, 0, 255) # Magenta
                label_prefix = "HB" # Handbag
            
            if is_loitering:
                color = (0, 0, 255) # Red Box
                
            # Draw Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
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

            elif is_loitering:
                label_bg_color = (0, 0, 255) # Red Label
                label = f"LOITERING {dwell_time:.1f}s"
            elif track_id is not None:
                 label = f"ID: {track_id}"
                 # Append Color Info if available
                 if det.get("shirt_color") and det["shirt_color"] != "Unknown":
                     s_c = det["shirt_color"]
                     # e.g. "ID: 10 (Black)"
                     label += f" ({s_c})"
            
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
        # Assuming 'frame' is the annotated_frame here
        if self.heatmap_enabled and self.heatmap_accumulator is not None:
            try:
                # 1. Blur the accumulator (on small map is consistent and fast)
                # Kernel size can be smaller now since map is smaller
                heatmap_blur = cv2.GaussianBlur(self.heatmap_accumulator, (21, 21), 0)
                
                # 2. Normalize to 0-255
                max_val = np.max(heatmap_blur)
                if max_val > 0:
                    heatmap_norm = (heatmap_blur / max_val * 255).astype(np.uint8)
                    
                    # 3. Apply Colormap
                    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
                    
                    # 4. Upscale to frame size
                    heatmap_resized = cv2.resize(heatmap_color, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
                    
                    # 5. Overlay
                    frame = cv2.addWeighted(frame, 0.6, heatmap_resized, 0.4, 0)
            except Exception as e:
                pass
                # print(f"Heatmap Error: {e}")

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
