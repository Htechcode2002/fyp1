import cv2
import time
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QImage, QPixmap
from src.core.detection import VideoDetector

class VideoThread(QThread):
    frame_signal = Signal(QPixmap)
    stats_signal = Signal(dict) # Emits counts/stats
    
    def __init__(self, source_path, resolution=None, tracker="bytetrack.yaml", location_name=None, video_id=None):
        super().__init__()
        self.source_path = source_path
        self.resolution = resolution
        self.tracking_enabled = True # Default On
        self.detection_enabled = True # Default On
        self._run_flag = True
        self.detector = VideoDetector(tracker=tracker, location_name=location_name, video_id=video_id)

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

    def set_pose_enabled(self, enabled):
        if self.detector:
            self.detector.set_pose_enabled(enabled)

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

        if str(src).isdigit():
            src = int(src)
        elif isinstance(src, str):
            if "youtube.com" in src or "youtu.be" in src:
                print("Resolving YouTube URL...")
                src = self.get_yt_url(src)
                print(f"Resolved to: {src}")
            else:
                # Normalize local file path
                import os
                if os.path.exists(src):
                    src = os.path.abspath(src)
                    print(f"DEBUG: Normalized local path: {src}")
            
        cap = cv2.VideoCapture(src)
        
        if not cap.isOpened():
             print(f"ERROR: Could not open video source: {src}")
             return
        else:
             backend = cap.getBackendName()
             w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
             h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
             fps = cap.get(cv2.CAP_PROP_FPS)
             frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
             print(f"DEBUG: Video opened. Backend: {backend}, Size: {w}x{h}, FPS: {fps}, Frames: {frames}")
             
             if w == 0 or h == 0:
                 print("ERROR: Video opened but dimensions are 0. Codec issue?")
                 return
        
        retries = 0
        while self._run_flag:
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
                # Run Detection?
                detections = []
                counts = {}
                
                if self.detection_enabled:
                    detections, counts = self.detector.detect(frame, tracking_enabled=self.tracking_enabled)
                else:
                    # If detection disabled, just use existing counts (don't update)
                    counts = self.detector.line_counts
                    
                frame = self.detector.annotate_frame(frame, detections, counts)
                
                # Emit stats
                self.stats_signal.emit(counts)
                
                # Convert to RGB
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                
                if self.resolution:
                    p = convert_to_Qt_format.scaled(self.resolution[0], self.resolution[1], Qt.KeepAspectRatio)
                else:
                    p = convert_to_Qt_format
                    
                self.frame_signal.emit(QPixmap.fromImage(p))

            # Limit FPS roughly
            time.sleep(0.03) 
            
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

