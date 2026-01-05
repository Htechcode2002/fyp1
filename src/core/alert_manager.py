"""
Alert Manager - Global cooldown tracker for Telegram alerts across multiple video sources.
Prevents duplicate alerts when the same person is detected by multiple cameras.
"""

import threading
import time


class AlertCooldownManager:
    """
    Singleton class to manage alert cooldowns across all video sources.
    Prevents duplicate alerts for the same event across multiple cameras.
    """
    _instance = None
    _lock = threading.Lock()

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

        # Global cooldown trackers
        # Key format: "{video_id}_{track_id}" or just "{track_id}" for global tracking
        self._fall_alerts = {}        # {key: last_alert_timestamp}
        self._loitering_alerts = {}   # {key: last_alert_timestamp}
        self._danger_alerts = {}      # {video_id: last_alert_timestamp}

        # Default cooldown periods (seconds)
        self.fall_cooldown = 60.0       # 1 minute
        self.loitering_cooldown = 60.0  # 1 minute
        self.danger_cooldown = 60.0     # 1 minute

        # Locks for thread-safe operations
        self._fall_lock = threading.Lock()
        self._loitering_lock = threading.Lock()
        self._danger_lock = threading.Lock()

        self._initialized = True
        print("[ALERT MANAGER] 🎯 Initialized global alert cooldown manager (Singleton)")

    def _make_key(self, video_id, track_id):
        """
        Create a unique key for tracking alerts.

        Args:
            video_id: Video source ID
            track_id: Person tracking ID

        Returns:
            Unique key string
        """
        if video_id and track_id is not None:
            return f"{video_id}_{track_id}"
        elif track_id is not None:
            return str(track_id)
        else:
            return None

    def can_send_fall_alert(self, video_id, track_id):
        """
        Check if a fall alert can be sent (cooldown check).

        Args:
            video_id: Video source ID
            track_id: Person tracking ID

        Returns:
            bool: True if alert can be sent, False if in cooldown
        """
        key = self._make_key(video_id, track_id)
        if key is None:
            return False

        with self._fall_lock:
            current_time = time.time()
            last_alert_time = self._fall_alerts.get(key, 0)
            time_since_last = current_time - last_alert_time

            if time_since_last >= self.fall_cooldown:
                return True
            else:
                remaining = self.fall_cooldown - time_since_last
                print(f"[ALERT MANAGER] ❌ Fall alert blocked for {key} - {remaining:.1f}s remaining")
                return False

    def mark_fall_alert_sent(self, video_id, track_id):
        """
        Mark that a fall alert has been sent.

        Args:
            video_id: Video source ID
            track_id: Person tracking ID
        """
        key = self._make_key(video_id, track_id)
        if key is None:
            return

        with self._fall_lock:
            current_time = time.time()
            self._fall_alerts[key] = current_time
            print(f"[ALERT MANAGER] ✅ Fall alert marked for {key} at {current_time:.2f}")

    def can_send_loitering_alert(self, video_id, track_id):
        """
        Check if a loitering alert can be sent (cooldown check).

        Args:
            video_id: Video source ID
            track_id: Person tracking ID

        Returns:
            bool: True if alert can be sent, False if in cooldown
        """
        key = self._make_key(video_id, track_id)
        if key is None:
            return False

        with self._loitering_lock:
            current_time = time.time()
            last_alert_time = self._loitering_alerts.get(key, 0)
            time_since_last = current_time - last_alert_time

            if time_since_last >= self.loitering_cooldown:
                return True
            else:
                remaining = self.loitering_cooldown - time_since_last
                print(f"[ALERT MANAGER] ❌ Loitering alert blocked for {key} - {remaining:.1f}s remaining")
                return False

    def mark_loitering_alert_sent(self, video_id, track_id):
        """
        Mark that a loitering alert has been sent.

        Args:
            video_id: Video source ID
            track_id: Person tracking ID
        """
        key = self._make_key(video_id, track_id)
        if key is None:
            return

        with self._loitering_lock:
            current_time = time.time()
            self._loitering_alerts[key] = current_time
            print(f"[ALERT MANAGER] ✅ Loitering alert marked for {key} at {current_time:.2f}")

    def can_send_danger_alert(self, video_id):
        """
        Check if a danger alert can be sent for a video source.

        Args:
            video_id: Video source ID

        Returns:
            bool: True if alert can be sent, False if in cooldown
        """
        if video_id is None:
            return False

        with self._danger_lock:
            current_time = time.time()
            last_alert_time = self._danger_alerts.get(video_id, 0)
            time_since_last = current_time - last_alert_time

            if time_since_last >= self.danger_cooldown:
                return True
            else:
                remaining = self.danger_cooldown - time_since_last
                print(f"[ALERT MANAGER] ❌ Danger alert blocked for {video_id} - {remaining:.1f}s remaining")
                return False

    def mark_danger_alert_sent(self, video_id):
        """
        Mark that a danger alert has been sent.

        Args:
            video_id: Video source ID
        """
        if video_id is None:
            return

        with self._danger_lock:
            current_time = time.time()
            self._danger_alerts[video_id] = current_time
            print(f"[ALERT MANAGER] ✅ Danger alert marked for {video_id} at {current_time:.2f}")

    def cleanup_old_entries(self, max_age=300):
        """
        Clean up entries older than max_age seconds to prevent memory growth.

        Args:
            max_age: Maximum age in seconds (default: 5 minutes)
        """
        current_time = time.time()
        cutoff_time = current_time - max_age

        with self._fall_lock:
            self._fall_alerts = {k: v for k, v in self._fall_alerts.items() if v > cutoff_time}

        with self._loitering_lock:
            self._loitering_alerts = {k: v for k, v in self._loitering_alerts.items() if v > cutoff_time}

        with self._danger_lock:
            self._danger_alerts = {k: v for k, v in self._danger_alerts.items() if v > cutoff_time}

        print(f"[ALERT MANAGER] 🧹 Cleaned up old entries (older than {max_age}s)")

    def get_stats(self):
        """Get current alert manager statistics."""
        with self._fall_lock, self._loitering_lock, self._danger_lock:
            return {
                "fall_alerts_tracked": len(self._fall_alerts),
                "loitering_alerts_tracked": len(self._loitering_alerts),
                "danger_alerts_tracked": len(self._danger_alerts),
                "fall_cooldown": self.fall_cooldown,
                "loitering_cooldown": self.loitering_cooldown,
                "danger_cooldown": self.danger_cooldown
            }

    def reset_all(self):
        """Reset all alert cooldowns (use with caution)."""
        with self._fall_lock, self._loitering_lock, self._danger_lock:
            self._fall_alerts.clear()
            self._loitering_alerts.clear()
            self._danger_alerts.clear()
            print("[ALERT MANAGER] 🔄 All alert cooldowns have been reset")


# Global singleton instance
_alert_manager_instance = AlertCooldownManager()


def get_alert_manager():
    """Get the global AlertCooldownManager singleton instance."""
    return _alert_manager_instance
