"""
Multi-Camera Cross-View Person Tracking Example

This example demonstrates how to use the cross-camera tracking system
to track people across multiple camera views.

Usage:
    python example_multi_camera.py

Requirements:
    - Multiple video files or camera streams
    - Updated detection.py with multi-camera support
    - ReID feature extractor
"""

import cv2
import threading
import queue
from src.core.detection import VideoDetector
from src.core.multi_camera_tracker import MultiCameraTracker


class MultiCameraSystem:
    """
    Multi-camera tracking system that processes multiple video streams
    and maintains global person IDs across cameras.
    """

    def __init__(self, video_sources, camera_ids=None):
        """
        Initialize multi-camera system.

        Args:
            video_sources: List of video file paths or camera indices
            camera_ids: List of camera identifiers (e.g., ['cam_0', 'cam_1'])
        """
        self.video_sources = video_sources
        self.num_cameras = len(video_sources)

        if camera_ids is None:
            self.camera_ids = [f"cam_{i}" for i in range(self.num_cameras)]
        else:
            self.camera_ids = camera_ids

        # Create shared multi-camera tracker
        self.multi_cam_tracker = MultiCameraTracker(
            similarity_threshold=0.6,  # Adjust based on testing
            max_time_gap=30  # 30 seconds max gap between sightings
        )

        # Create detector for each camera
        self.detectors = []
        for i, camera_id in enumerate(self.camera_ids):
            detector = VideoDetector(
                tracker="trackers/botsort_reid.yaml",
                location_name=f"Camera {i+1}",
                camera_id=camera_id,
                multi_cam_tracker=self.multi_cam_tracker
            )
            self.detectors.append(detector)

        # Video captures
        self.captures = []
        for source in video_sources:
            cap = cv2.VideoCapture(source)
            self.captures.append(cap)

        # Frame queues for each camera
        self.frame_queues = [queue.Queue(maxsize=2) for _ in range(self.num_cameras)]

        # Running flag
        self.running = False

    def capture_thread(self, camera_idx):
        """
        Thread function to capture frames from a camera.

        Args:
            camera_idx: Index of the camera
        """
        cap = self.captures[camera_idx]
        frame_queue = self.frame_queues[camera_idx]

        while self.running:
            ret, frame = cap.read()
            if not ret:
                # Loop video or restart
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Put frame in queue (drop old frames if queue is full)
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass

            frame_queue.put(frame)

    def process_thread(self, camera_idx):
        """
        Thread function to process frames from a camera.

        Args:
            camera_idx: Index of the camera
        """
        detector = self.detectors[camera_idx]
        frame_queue = self.frame_queues[camera_idx]
        camera_id = self.camera_ids[camera_idx]

        while self.running:
            try:
                frame = frame_queue.get(timeout=1)
            except queue.Empty:
                continue

            # Run detection with tracking
            detections, counts = detector.detect(frame, tracking_enabled=True)

            # Annotate frame
            annotated_frame = detector.annotate_frame(frame.copy(), detections, counts)

            # Display camera view
            window_name = f"Camera {camera_idx + 1} - {camera_id}"
            cv2.imshow(window_name, annotated_frame)

    def run(self):
        """
        Run the multi-camera tracking system.
        """
        self.running = True

        # Start capture threads
        capture_threads = []
        for i in range(self.num_cameras):
            t = threading.Thread(target=self.capture_thread, args=(i,))
            t.daemon = True
            t.start()
            capture_threads.append(t)

        # Start processing threads
        process_threads = []
        for i in range(self.num_cameras):
            t = threading.Thread(target=self.process_thread, args=(i,))
            t.daemon = True
            t.start()
            process_threads.append(t)

        print("\n=== Multi-Camera Tracking System ===")
        print(f"Tracking across {self.num_cameras} cameras")
        print("Press 'q' to quit")
        print("Press 's' to show statistics")
        print("\n")

        # Main loop
        while self.running:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                self.running = False
                break

            elif key == ord('s'):
                # Show statistics
                stats = self.multi_cam_tracker.get_statistics()
                print("\n=== Cross-Camera Tracking Statistics ===")
                print(f"Total Global Persons: {stats['total_global_persons']}")
                print(f"Active Cameras: {stats['active_cameras']}")
                print(f"Total Mappings: {stats['total_mappings']}")
                print()

                # Show some person details
                all_persons = self.multi_cam_tracker.get_all_persons()
                print("Recent Global IDs:")
                for gid, data in list(all_persons.items())[:10]:
                    cameras = list(data['camera_tracks'].keys())
                    color = data.get('color', 'Unknown')
                    print(f"  G{gid}: Seen in {cameras}, Color: {color}")
                print()

        # Cleanup
        self.cleanup()

    def cleanup(self):
        """
        Clean up resources.
        """
        self.running = False

        # Release captures
        for cap in self.captures:
            cap.release()

        # Close windows
        cv2.destroyAllWindows()


def main():
    """
    Example usage of multi-camera tracking system.
    """

    # Example 1: Two video files
    video_sources = [
        "videos/camera1.mp4",
        "videos/camera2.mp4"
    ]

    # Example 2: Three camera streams (uncomment to use)
    # video_sources = [0, 1, 2]  # Camera indices

    # Example 3: Mix of file and camera (uncomment to use)
    # video_sources = [
    #     "videos/entrance.mp4",
    #     "videos/corridor.mp4",
    #     "videos/exit.mp4",
    #     0  # Live camera
    # ]

    camera_ids = [f"cam_{i}" for i in range(len(video_sources))]

    # Create and run system
    system = MultiCameraSystem(video_sources, camera_ids)

    try:
        system.run()
    except KeyboardInterrupt:
        print("\nStopping...")
        system.cleanup()


if __name__ == "__main__":
    main()
