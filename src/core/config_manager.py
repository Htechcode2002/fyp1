import json
import os
import copy

CONFIG_FILE = "config.json"

DEFAULT_CONFIG = {
    "db": {
        "host": "localhost",
        "port": 3306,
        "user": "root"
    },
    "yolo": {
        "model_path": "models/yolo12n.pt"
    },
    "thresholds": {
        "warning": 20,
        "critical": 50
    },
    "video_sources": []
}

class ConfigManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance.config = copy.deepcopy(DEFAULT_CONFIG)
            cls._instance.load_config()
        return cls._instance

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    data = json.load(f)
                    # Reset and Merge with default to ensure all keys exist
                    self.config = copy.deepcopy(DEFAULT_CONFIG)
                    self.config.update(data)
            except Exception as e:
                print(f"Error loading config: {e}")
        return self.config

    def save_config(self):
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            print(f"Error saving config: {e}")

    def get(self, key, default=None):
        return self.config.get(key, default)

    def set(self, key, value):
        self.config[key] = value
        self.save_config()

    def get_active_count(self):
        """Returns number of active video sources for dynamic optimization"""
        return len(self.config.get("video_sources", []))
