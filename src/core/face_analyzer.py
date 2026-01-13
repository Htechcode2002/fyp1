import os
import cv2
import numpy as np
import threading

# --- DLL FIX FOR TENSORRT ON WINDOWS ---
if os.name == 'nt':
    try:
        # More robust path detection
        import tensorrt_libs
        dll_dir = os.path.dirname(tensorrt_libs.__file__)
        if os.path.exists(dll_dir):
            os.add_dll_directory(dll_dir)
            if dll_dir not in os.environ['PATH']:
                os.environ['PATH'] = dll_dir + os.pathsep + os.environ['PATH']
            print(f"[FACE] 🛠️ TensorRT DLLs active: {dll_dir}")
    except Exception:
        pass

try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False

class FaceAnalyzer:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(FaceAnalyzer, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name='buffalo_l', ctx_id=0, det_size=(640, 640), enabled=True):
        if getattr(self, '_initialized', False):
            return

        self.enabled = enabled
        self.app = None
        self.yolo_face_model = None # Placeholder
        self._lock = threading.Lock()
        
        if self.enabled:
            try:
                print(f"[FACE] 🚀 Initializing Global Mode (InsightFace: {model_name})...")
                # Use CUDA for stability - ROI optimization already provides the speed we need
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                self.app = FaceAnalysis(name=model_name, providers=providers)
                self.app.prepare(ctx_id=ctx_id, det_size=det_size)
                print("[FACE] ✅ System ready.")
            except Exception as e:
                print(f"[FACE] ❌ Model load error: {e}")
                self.enabled = False

        self._initialized = True

    def set_yolo_face_model(self, model):
        """Placeholder for compatibility, no longer needed in Global Mode"""
        self.yolo_face_model = model

    def analyze_faces_global(self, frame, roi_boxes=None):
        """
        Targeted ROI Inference. 
        Uses native resolution crops for maximum clarity on CUDA.
        """
        if not self.enabled or self.app is None:
            return []

        try:
            results = []
            
            # --- ROI MODE: Targeted Inference (Native Proportions) ---
            if roi_boxes is not None and len(roi_boxes) > 0:
                h, w = frame.shape[:2]
                # Limit to 5 people per frame, prioritized by size
                for box in roi_boxes[:5]:
                    x1, y1, x2, y2 = box
                    # Add 25% margin for better face context
                    mw = int((x2 - x1) * 0.25)
                    mh = int((y2 - y1) * 0.25)
                    
                    cx1 = max(0, int(x1 - mw))
                    cy1 = max(0, int(y1 - mh))
                    cx2 = min(w, int(x2 + mw))
                    cy2 = min(h, int(y2 + mh))
                    
                    crop = frame[cy1:cy2, cx1:cx2]
                    if crop.size == 0: continue
                    
                    # IMPORTANT: Do NOT manually resize here. 
                    # app.get() handles its own det_size and maintains aspect ratio better.
                    # Manually resizing to 640x640 stretches the person and distorts the face.
                    with self._lock:
                        faces = self.app.get(crop)
                    
                    for face in faces:
                        fx1, fy1, fx2, fy2 = face.bbox
                        # Simple coordinate reconstruction (Offset only)
                        results.append({
                            'bbox': [
                                int(fx1) + cx1, 
                                int(fy1) + cy1, 
                                int(fx2) + cx1, 
                                int(fy2) + cy1
                            ],
                            'gender': 'Male' if face.gender == 1 else 'Female',
                            'age': int(face.age)
                        })
                return results

            # --- GLOBAL MODE: Full frame fallback (Native Res) ---
            with self._lock:
                faces = self.app.get(frame)
            
            for face in faces:
                x1, y1, x2, y2 = face.bbox
                results.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'gender': 'Male' if face.gender == 1 else 'Female',
                    'age': int(face.age)
                })
            return results
        except Exception as e:
            print(f"[FACE] ❌ Inference error: {e}")
            return []

    def draw_face_info(self, frame, face_info):
        """Premium 'Corner-Only' UI style for a modern look"""
        for face in face_info:
            x1, y1, x2, y2 = face['bbox']
            gender = face.get('gender', 'Face')
            age = face.get('age', '')
            
            # Neon Cyan/Yellow color
            color = (0, 255, 255)
            l = 15 # Length of corner lines
            t = 2  # Thickness
            
            # Draw premium corner brackets instead of a full box
            cv2.line(frame, (x1, y1), (x1 + l, y1), color, t)
            cv2.line(frame, (x1, y1), (x1, y1 + l), color, t)
            
            cv2.line(frame, (x2, y1), (x2 - l, y1), color, t)
            cv2.line(frame, (x2, y1), (x2, y1 + l), color, t)
            
            cv2.line(frame, (x1, y2), (x1 + l, y2), color, t)
            cv2.line(frame, (x1, y2), (x1, y2 - l), color, t)
            
            cv2.line(frame, (x2, y2), (x2 - l, y2), color, t)
            cv2.line(frame, (x2, y2), (x2, y2 - l), color, t)

            # Center Dot
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(frame, (cx, cy), 3, color, -1)
            
            # Label
            label = f"{gender} {age}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(label, font, 0.4, 1)
            
            # Background for label (above the box)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 10, y1), color, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 6), font, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
            
        return frame
