import cv2
import time
import os
import threading
import datetime
import traceback
from queue import Queue
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QImage, QPixmap
from src.core.detection import VideoDetector

class AsyncVideoWriter(threading.Thread):
    """Background thread for non-blocking video writing"""
    def __init__(self, output_path, fps, size):
        super().__init__()
        self.output_path = output_path
        self.fps = fps
        self.size = size
        self.queue = Queue(maxsize=150) # Buffer for ~5-10 seconds of delay
        self.running = True
        self.writer = None
        
    def run(self):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, self.size)
        
        while self.running or not self.queue.empty():
            try:
                # Use a small timeout so we can check self.running periodically
                frame = self.queue.get(timeout=0.1)
                if frame is not None:
                    self.writer.write(frame)
                self.queue.task_done()
            except:
                # Queue empty, just wait
                time.sleep(0.005)
                
        if self.writer:
            self.writer.release()
            
    def add_frame(self, frame):
        """Add frame to queue. Blocks if recording must keep up without dropping frames."""
        if not self.running:
            return False
            
        try:
            # We use a timeout to prevent permanent deadlocks, 
            # but effectively block to ensure no frames are dropped during recording.
            self.queue.put(frame.copy(), block=True, timeout=0.5)
            return True
        except:
            # If still full after 0.5s, something is seriously wrong with disk/CPU speed
            print("[RECORD] ⚠️ Warning: Buffer full, 1 frame dropped (Disk too slow?)")
            return False

    def stop(self):
        self.running = False
        # Add a sentinel to wake up the thread and finish the queue
        if self.queue.empty():
            self.queue.put(None) 
        self.join()

class VideoThread(QThread):
    frame_signal = Signal(QImage)
    stats_signal = Signal(dict) # Emits counts/stats
    
    def __init__(self, source_path, resolution=None, tracker="bytetrack.yaml", location_name=None, video_id=None, danger_threshold=100, loitering_threshold=5.0, fall_threshold=2.0):
        super().__init__()
        self.source_path = source_path
        self.resolution = resolution
        self.tracking_enabled = True # Default On
        self.detection_enabled = True # Default On
        self._run_flag = True
        self.detector = VideoDetector(tracker=tracker, location_name=location_name, video_id=video_id, danger_threshold=danger_threshold, loitering_threshold=loitering_threshold, fall_threshold=fall_threshold)
        
        # Recording State
        self.is_recording = False
        self.async_writer = None
        self.recorded_frames = 0
        self.recording_start_time = 0
        self.segment_limit = 300 # 5 minutes per segment
        self.recording_output_path = None

    def set_tracking(self, enabled):
        self.tracking_enabled = enabled

    def set_detection(self, enabled):
        self.detection_enabled = enabled
        
    def set_lines(self, lines):
        if self.detector:
            self.detector.set_lines(lines)

    def set_zones(self, zones):
        if self.detector:
            self.detector.set_zones(zones)
            
    def set_heatmap(self, enabled):
        if self.detector:
            self.detector.set_heatmap(enabled)

    def set_fall_detection(self, enabled):
        if self.detector:
            self.detector.set_fall_detection(enabled)

    def set_face_analysis_enabled(self, enabled):
        if self.detector:
            self.detector.set_face_analysis_enabled(enabled)

    def set_mask_detection(self, enabled):
        if self.detector:
            self.detector.set_mask_detection(enabled)

    def set_display_mode(self, mode):
        """Set display mode: 'dot' or 'box'"""
        if self.detector:
            self.detector.set_display_mode(mode)

    def start_recording(self, output_path):
        """Initialize video writer for recording"""
        self.recording_output_path = output_path
        self.is_recording = True
        self.recording_start_time = time.time()
        
        # Synchronize with detector for visual indicator
        if self.detector:
            self.detector.set_recording_state(True)
            
        # Writer will be initialized on first frame
        print(f"[RECORD] 🔴 Recording sequence initiated: {output_path}")

    def stop_recording(self):
        """Finalize and close video writer"""
        self.is_recording = False
        
        # Synchronize with detector
        if self.detector:
            self.detector.set_recording_state(False)
            
        if self.async_writer:
            self.async_writer.stop()
            self.async_writer = None
        
        if self.recorded_frames > 0:
            print(f"[RECORD] ✅ Recording saved. ({self.recorded_frames} frames)")
        self.recorded_frames = 0

    def get_yt_url(self, url):
        import yt_dlp
        ydl_opts = {'format': 'best', 'quiet': True}
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                return info.get('url', url)
        except Exception as e:
            print(f"Error resolving YouTube URL: {e}")
            return url
        
    def run(self):
        # Handle numeric ID for webcam or string for file/url
        src = self.source_path
        print(f"DEBUG: VideoThread starting with source: {src} (Type: {type(src)})")
        
        if src is None or src == "":
            print("ERROR: Video source is empty!")
            return

        if str(src).lower() == "auto":
            print("[VIDEO] 🔍 Auto-detecting best available camera...")
            found_src = None
            for i in range(5): # Try indices 0 to 4
                temp_cap = cv2.VideoCapture(i)
                if temp_cap.isOpened():
                    ret, _ = temp_cap.read()
                    if ret:
                        print(f"[VIDEO] ✅ Auto-detected working camera at index: {i}")
                        found_src = i
                        temp_cap.release()
                        break
                temp_cap.release()
            
            if found_src is not None:
                src = found_src
            else:
                print("[VIDEO] ❌ No working cameras found during auto-detect.")
                return

        elif str(src).isdigit():
            src = int(src)
        elif isinstance(src, str):
            if "youtube.com" in src or "youtu.be" in src:
                print("Resolving YouTube URL...")
                src = self.get_yt_url(src)
                print(f"Resolved to: {src}")
            else:
                if os.path.exists(src):
                    src = os.path.abspath(src)
                    print(f"DEBUG: Normalized local path: {src}")
            
        cap = cv2.VideoCapture(src)

        # CRITICAL FIX: Set low-latency mode for RTSP streams to prevent delay accumulation
        if isinstance(src, str) and src.startswith("rtsp://"):
            print("[VIDEO] 🔧 Detected RTSP stream - Applying low-latency optimizations...")
            # Minimize buffering for real-time streams
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Only buffer 1 frame (critical for low latency)
            # Use TCP instead of UDP for more stable connection (optional, can be removed if causes issues)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)  # 5 second timeout
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)  # 5 second read timeout

        if not cap.isOpened():
             if isinstance(src, int):
                 print(f"[VIDEO] ⚠️ Camera index {src} failed. Falling back to Auto-Detect...")
                 found_src = None
                 for i in range(5):
                     temp_cap = cv2.VideoCapture(i)
                     if temp_cap.isOpened():
                         ret, _ = temp_cap.read()
                         if ret:
                             found_src = i
                             temp_cap.release()
                             break
                     temp_cap.release()
                 
                 if found_src is not None:
                     print(f"[VIDEO] ✅ Fallback successful. Using index: {found_src}")
                     src = found_src
                     cap = cv2.VideoCapture(src)
                 else:
                     print(f"ERROR: Could not open video source: {src} and no fallbacks found.")
                     return
             else:
                 print(f"ERROR: Could not open video source: {src}")
                 return
        else:
             backend = cap.getBackendName()
             w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
             h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
             fps = cap.get(cv2.CAP_PROP_FPS)
             self.fps = fps if fps > 0 else 25 # Fallback to 25 if data missing
             frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
             print(f"DEBUG: Video opened. Backend: {backend}, Size: {w}x{h}, FPS: {self.fps}, Frames: {frames}")
             
             if w == 0 or h == 0:
                 print("ERROR: Video opened but dimensions are 0. Codec issue?")
                 return
        
        retries = 0
        is_rtsp_stream = isinstance(src, str) and src.startswith("rtsp://")
        
        # Initialize timing
        last_frame_time = time.time()

        while self._run_flag:
            current_loop_start = time.time()
            # CRITICAL FIX: For RTSP streams, grab latest frame and discard buffered frames
            # This prevents delay accumulation
            if is_rtsp_stream:
                # Grab multiple frames to flush buffer and get the latest one
                for _ in range(2):  # Read and discard 1 old frame, keep the latest
                    cap.grab()

            ret, frame = cap.read()
            if not ret:
                # Check if it's a local file (Loop it) or a stream (Retry)
                is_local_file = isinstance(src, str) and not (src.startswith("http") or "youtube" in src)
                
                if is_local_file:
                    print("Video ended (or failed read), looping back to start...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    if self.detector:
                        self.detector.reset_analytics()
                    time.sleep(0.1) # Brief pause
                    continue
                else:
                    # Stream Error Logic
                    if retries < 5:
                        print(f"WARNING: Failed to read frame from stream. Retry {retries+1}/5")
                        retries += 1
                        time.sleep(1)
                        continue
                    else:
                        print("ERROR: Failed to read frame after multiple retries.")
                        break
            else:
                retries = 0 # Reset on success
            
            if ret:
                try:
                    # Run Detection?
                    detections = []
                    counts = {}
                    
                    if self.detection_enabled:
                        detections, counts = self.detector.detect(frame, tracking_enabled=self.tracking_enabled)
                    else:
                        # If detection disabled, just use existing counts (don't update)
                        counts = self.detector.line_counts
                        
                    frame = self.detector.annotate_frame(frame, detections, counts)
                    
                    # --- ASYNC RECORDING LOGIC ---
                    if self.is_recording:
                        current_time_rec = time.time()
                        
                        # Handle Segmentation (Rotation)
                        if self.recorded_frames > 0 and (current_time_rec - self.recording_start_time) > self.segment_limit:
                            print("[RECORD] 🔄 Reached segment limit. Rotating file...")
                            
                            # 1. Close current writer
                            old_path = self.recording_output_path
                            start_ts = self.recording_start_time
                            if self.async_writer:
                                self.async_writer.stop()
                                self.async_writer = None
                            
                            # 2. Rename the COMPLETED segment with duration (Professional touch)
                            try:
                                end_ts = current_time_rec
                                dt_start = datetime.datetime.fromtimestamp(start_ts)
                                dt_end = datetime.datetime.fromtimestamp(end_ts)
                                
                                new_name = f"{dt_start.strftime('%Y%m%d_%H%M%S')}__TO__{dt_end.strftime('%H%M%S')}.mp4"
                                new_path = os.path.join(os.path.dirname(old_path), new_name)
                                
                                # Wait a tiny bit for file closure
                                time.sleep(0.1)
                                if os.path.exists(old_path):
                                    os.rename(old_path, new_path)
                                    # Also move thumbnail
                                    old_thumb = old_path.replace(".mp4", "_thumb.jpg")
                                    new_thumb = new_path.replace(".mp4", "_thumb.jpg")
                                    if os.path.exists(old_thumb):
                                        os.rename(old_thumb, new_thumb)
                                    print(f"[RECORD] 📝 Segment finalized: {new_name}")
                            except Exception as e:
                                print(f"[RECORD] ⚠️ Failed to rename segment: {e}")

                            # 3. Setup new path for NEXT segment
                            dir_name = os.path.dirname(old_path)
                            timestamp = time.strftime("%Y%m%d_%H%M%S")
                            self.recording_output_path = os.path.abspath(os.path.join(dir_name, f"{timestamp}_REC.mp4"))
                            self.recording_start_time = current_time_rec
                            self.recorded_frames = 0 # Reset frame count for new segment
                            print(f"[RECORD] 🔴 New segment started: {self.recording_output_path}")

                        # Initialize Writer & Thumbnail
                        if self.async_writer is None:
                            h_rec, w_rec = frame.shape[:2]
                            self.async_writer = AsyncVideoWriter(self.recording_output_path, self.fps, (w_rec, h_rec))
                            self.async_writer.start()
                            
                            # Save Thumbnail (Optimization 1)
                            try:
                                thumb_path = self.recording_output_path.replace(".mp4", "_thumb.jpg")
                                # Create small thumbnail
                                thumb_w = 400
                                thumb_h = int(h_rec * (thumb_w / w_rec))
                                thumb_frame = cv2.resize(frame, (thumb_w, thumb_h))
                                cv2.imwrite(thumb_path, thumb_frame)
                                print(f"[RECORD] 🖼️ Thumbnail generated: {os.path.basename(thumb_path)}")
                            except Exception as e:
                                print(f"[RECORD] ⚠️ Thumbnail error: {e}")
                        
                        # Queue Frame (Async Optimization 2)
                        if self.async_writer:
                            if self.async_writer.add_frame(frame):
                                self.recorded_frames += 1

                    
                    # Emit stats (Throttled to once every 15 frames or on change)
                    if not hasattr(self, '_last_counts'):
                        self._last_counts = {}
                    
                    # Check if counts significantly changed or enough time passed
                    counts_changed = counts != self._last_counts
                    if counts_changed or (self.detector.frame_count % 15 == 0):
                        self.stats_signal.emit(counts)
                        self._last_counts = counts.copy()
                    
                    # Convert to RGB
                    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_image.shape
                    bytes_per_line = ch * w
                    
                    # CRITICAL FIX: QImage with numpy data can crash if numpy array is garbage collected
                    # Must use .copy() to create a deep copy that Qt owns
                    convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
                    
                    if self.resolution:
                        p = convert_to_Qt_format.scaled(self.resolution[0], self.resolution[1], Qt.KeepAspectRatio)
                    else:
                        p = convert_to_Qt_format
                        
                    # Performance Lock: Ensure we don't spam signals too fast for the UI
                    if not hasattr(self, '_last_emit_time'):
                        self._last_emit_time = 0
                    
                    
                    current_time = time.time()
                    if (current_time - self._last_emit_time) >= 0.038: # Max ~26 FPS for UI stability
                        self.frame_signal.emit(convert_to_Qt_format)
                        self._last_emit_time = current_time
                    else:
                        # Skip signal to save UI processing, object will be cleaned by GC
                        pass
                    
                    # Force local cleanup
                    del rgb_image
                    
                except Exception as e:
                    print(f"[VIDEO THREAD] ❌ Error processing frame: {e}")
                    traceback.print_exc()

            # Sync with video FPS to prevent "fast-forward" effect
            # For RTSP streams, minimize sleep to reduce latency
            if is_rtsp_stream:
                # For real-time streams, only sleep minimally to allow processing
                time.sleep(0.001)  # 1ms sleep to prevent CPU spinning
            else:
                # For local files, sync with video FPS
                fps = self.fps if hasattr(self, 'fps') and self.fps > 0 else 25
                delay = 1.0 / fps

                # Adjust sleep to account for processing time
                elapsed = time.time() - current_loop_start
                sleep_time = max(0.001, delay - elapsed)
                time.sleep(sleep_time) 
            
        cap.release()

    def stop(self):
        self._run_flag = False
        self.stop_recording() # Ensure writer is released
        self.wait()

