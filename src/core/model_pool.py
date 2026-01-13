"""
Model Pool - Singleton pattern for sharing YOLO models across multiple video sources.
This significantly reduces GPU memory usage and improves performance.
"""

from ultralytics import YOLO
import threading
import torch
import numpy as np


class ModelPool:
    """
    Singleton class to manage shared YOLO model instances.
    All video sources share the same model instances to save GPU memory.
    """
    _instance = None
    _lock = threading.Lock()
    _model_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        # Shared model instances
        self._main_model = None
        self._fall_model = None
        self._mask_model = None
        self._pose_model = None
        self._face_model = None

        # Reference counters to track usage
        self._main_refs = 0
        self._fall_refs = 0
        self._mask_refs = 0
        self._pose_refs = 0
        self._face_refs = 0

        # Locks for thread-safe model loading
        self._main_lock = threading.Lock()
        self._fall_lock = threading.Lock()
        self._mask_lock = threading.Lock()
        self._pose_lock = threading.Lock()
        self._face_lock = threading.Lock()

        self._initialized = True
        print("[MODEL POOL] 🎯 Initialized shared model pool (Singleton)")

    def get_main_model(self, model_path="models/yolov8n.pt"):
        """
        Get shared main YOLO detection model.

        Args:
            model_path: Path to YOLO model weights

        Returns:
            Shared YOLO model instance
        """
        with self._main_lock:
            if self._main_model is None:
                print(f"[MODEL POOL] 📥 Loading main YOLO model: {model_path}")
                self._main_model = YOLO(model_path)
                print("[MODEL POOL] ✅ Main YOLO model loaded")

                # CRITICAL FIX: Warmup model to trigger fuse() in a thread-safe context
                # This prevents multiple threads from trying to fuse simultaneously
                print("[MODEL POOL] 🔥 Warming up model (running first inference)...")
                dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
                self._main_model.predict(dummy_frame, verbose=False, imgsz=960, half=torch.cuda.is_available())
                print("[MODEL POOL] ✅ Model warmup complete")

            self._main_refs += 1
            print(f"[MODEL POOL] 🔗 Main model reference count: {self._main_refs}")
            return self._main_model

    def release_main_model(self):
        """Release reference to main model."""
        with self._main_lock:
            if self._main_refs > 0:
                self._main_refs -= 1
                print(f"[MODEL POOL] 🔓 Main model reference count: {self._main_refs}")

    def get_fall_model(self, model_path="models/best.pt"):
        """
        Get shared fall detection model.

        Args:
            model_path: Path to fall detection model weights

        Returns:
            Shared fall detection model instance
        """
        with self._fall_lock:
            if self._fall_model is None:
                print(f"[MODEL POOL] 📥 Loading fall detection model: {model_path}")
                try:
                    self._fall_model = YOLO(model_path)
                    print("[MODEL POOL] ✅ Fall detection model loaded")

                    # Warmup to trigger fuse
                    print("[MODEL POOL] 🔥 Warming up fall model...")
                    dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
                    self._fall_model.predict(dummy_frame, verbose=False, imgsz=960, half=torch.cuda.is_available())
                    print("[MODEL POOL] ✅ Fall model warmup complete")
                except Exception as e:
                    print(f"[MODEL POOL] ❌ Failed to load fall model: {e}")
                    return None

            self._fall_refs += 1
            print(f"[MODEL POOL] 🔗 Fall model reference count: {self._fall_refs}")
            return self._fall_model

    def release_fall_model(self):
        """Release reference to fall detection model."""
        with self._fall_lock:
            if self._fall_refs > 0:
                self._fall_refs -= 1
                print(f"[MODEL POOL] 🔓 Fall model reference count: {self._fall_refs}")

    def get_mask_model(self, model_path="models/mask_detection.pt"):
        """
        Get shared mask detection model.

        Args:
            model_path: Path to mask detection model weights

        Returns:
            Shared mask detection model instance
        """
        with self._mask_lock:
            if self._mask_model is None:
                print(f"[MODEL POOL] 📥 Loading mask detection model: {model_path}")
                try:
                    self._mask_model = YOLO(model_path)
                    print("[MODEL POOL] ✅ Mask detection model loaded")

                    # Warmup to trigger fuse
                    print("[MODEL POOL] 🔥 Warming up mask model...")
                    dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
                    self._mask_model.predict(dummy_frame, verbose=False, imgsz=960, half=torch.cuda.is_available())
                    print("[MODEL POOL] ✅ Mask model warmup complete")
                except Exception as e:
                    print(f"[MODEL POOL] ❌ Failed to load mask model: {e}")
                    return None

            self._mask_refs += 1
            print(f"[MODEL POOL] 🔗 Mask model reference count: {self._mask_refs}")
            return self._mask_model

    def release_mask_model(self):
        """Release reference to mask detection model."""
        with self._mask_lock:
            if self._mask_refs > 0:
                self._mask_refs -= 1
                print(f"[MODEL POOL] 🔓 Mask model reference count: {self._mask_refs}")

    def get_pose_model(self, model_path="models/yolov8n-pose.pt"):
        """
        Get shared pose estimation model.

        Args:
            model_path: Path to pose model weights

        Returns:
            Shared pose model instance
        """
        with self._pose_lock:
            if self._pose_model is None:
                print(f"[MODEL POOL] 📥 Loading pose model: {model_path}")
                try:
                    self._pose_model = YOLO(model_path)
                    print("[MODEL POOL] ✅ Pose model loaded")

                    # Warmup to trigger fuse
                    print("[MODEL POOL] 🔥 Warming up pose model...")
                    dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
                    self._pose_model.predict(dummy_frame, verbose=False, imgsz=640, half=torch.cuda.is_available())
                    print("[MODEL POOL] ✅ Pose model warmup complete")
                except Exception as e:
                    print(f"[MODEL POOL] ❌ Failed to load pose model: {e}")
                    return None

            self._pose_refs += 1
            print(f"[MODEL POOL] 🔗 Pose model reference count: {self._pose_refs}")
            return self._pose_model

    def release_pose_model(self):
        """Release reference to pose model."""
        with self._pose_lock:
            if self._pose_refs > 0:
                self._pose_refs -= 1
                print(f"[MODEL POOL] 🔓 Pose model reference count: {self._pose_refs}")

    def get_face_model(self, model_path="models/face/yolov8n-face.pt"):
        """
        Get shared YOLO face detection model.
        """
        with self._face_lock:
            if self._face_model is None:
                print(f"[MODEL POOL] 📥 Loading face detection model: {model_path}")
                try:
                    self._face_model = YOLO(model_path)
                    print("[MODEL POOL] ✅ Face model loaded")

                    # Warmup
                    print("[MODEL POOL] 🔥 Warming up face model...")
                    dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
                    self._face_model.predict(dummy_frame, verbose=False, imgsz=640, half=torch.cuda.is_available())
                    print("[MODEL POOL] ✅ Face model warmup complete")
                except Exception as e:
                    print(f"[MODEL POOL] ❌ Failed to load face model: {e}")
                    return None

            self._face_refs += 1
            print(f"[MODEL POOL] 🔗 Face model reference count: {self._face_refs}")
            return self._face_model

    def release_face_model(self):
        """Release reference to face model."""
        with self._face_lock:
            if self._face_refs > 0:
                self._face_refs -= 1
                print(f"[MODEL POOL] 🔓 Face model reference count: {self._face_refs}")

    def get_stats(self):
        """Get current model pool statistics."""
        return {
            "main_model_loaded": self._main_model is not None,
            "main_refs": self._main_refs,
            "fall_model_loaded": self._fall_model is not None,
            "fall_refs": self._fall_refs,
            "mask_model_loaded": self._mask_model is not None,
            "mask_refs": self._mask_refs,
            "pose_model_loaded": self._pose_model is not None,
            "pose_refs": self._pose_refs,
            "face_model_loaded": self._face_model is not None,
            "face_refs": self._face_refs
        }

    def cleanup_unused(self):
        """
        Clean up models with zero references.
        Call this periodically to free GPU memory.
        """
        cleaned = []

        with self._main_lock:
            if self._main_refs == 0 and self._main_model is not None:
                self._main_model = None
                cleaned.append("main")

        with self._fall_lock:
            if self._fall_refs == 0 and self._fall_model is not None:
                self._fall_model = None
                cleaned.append("fall")

        with self._mask_lock:
            if self._mask_refs == 0 and self._mask_model is not None:
                self._mask_model = None
                cleaned.append("mask")

        with self._pose_lock:
            if self._pose_refs == 0 and self._pose_model is not None:
                self._pose_model = None
                cleaned.append("pose")

        with self._face_lock:
            if self._face_refs == 0 and self._face_model is not None:
                self._face_model = None
                cleaned.append("face")

        if cleaned:
            print(f"[MODEL POOL] 🧹 Cleaned up unused models: {', '.join(cleaned)}")

        return cleaned


# Global singleton instance
_model_pool_instance = ModelPool()


def get_model_pool():
    """Get the global ModelPool singleton instance."""
    return _model_pool_instance
