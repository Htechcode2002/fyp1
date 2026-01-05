"""
Telegram Notifier for sending alerts when danger threshold is exceeded.
"""

import requests
import cv2
import os
import time
from datetime import datetime
from pathlib import Path
from threading import Thread


class TelegramNotifier:
    """Send notifications and screenshots via Telegram bot."""

    def __init__(self, bot_token=None, chat_id=None):
        """
        Initialize Telegram notifier.

        Args:
            bot_token: Telegram bot token (from BotFather)
            chat_id: Your Telegram chat ID
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = False
        self.last_alert_time = 0
        self.alert_cooldown = 60  # Minimum 60 seconds between alerts (avoid spam)

        # Check if credentials are provided
        if bot_token and chat_id:
            self.enabled = True
            print(f"✅ Telegram notifier enabled for chat ID: {chat_id}")
        else:
            print("⚠️ Telegram notifier disabled (missing credentials)")

        # Create screenshots directory
        self.screenshots_dir = Path("screenshots")
        self.screenshots_dir.mkdir(exist_ok=True)

    def send_message(self, message):
        """
        Send a text message via Telegram.

        Args:
            message: Text message to send

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.enabled:
            return False

        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            data = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "HTML"
            }

            response = requests.post(url, data=data, timeout=10)
            return response.status_code == 200

        except Exception as e:
            print(f"❌ Failed to send Telegram message: {e}")
            return False

    def send_photo(self, image_path, caption=""):
        """
        Send a photo via Telegram.

        Args:
            image_path: Path to the image file
            caption: Optional caption for the photo

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.enabled:
            return False

        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"

            with open(image_path, 'rb') as photo:
                files = {'photo': photo}
                data = {
                    'chat_id': self.chat_id,
                    'caption': caption,
                    'parse_mode': 'HTML'
                }

                response = requests.post(url, files=files, data=data, timeout=30)
                return response.status_code == 200

        except Exception as e:
            print(f"❌ Failed to send Telegram photo: {e}")
            return False

    def send_alert(self, message, image_path=None):
        """
        Send a general alert with optional image.

        Args:
            message: Alert message text
            image_path: Optional path to image file

        Returns:
            bool: True if alert sent successfully
        """
        if not self.enabled:
            return False

        try:
            if image_path and os.path.exists(image_path):
                # Send with photo
                return self.send_photo(image_path, caption=message)
            else:
                # Send text only
                return self.send_message(message)

        except Exception as e:
            print(f"❌ Failed to send alert: {e}")
            return False

    def send_alert_async(self, message, image_path=None):
        """
        Send alert asynchronously in background thread to avoid blocking video processing.

        Args:
            message: Alert message text
            image_path: Optional path to image file
        """
        if not self.enabled:
            return

        def _send():
            try:
                if image_path and os.path.exists(image_path):
                    self.send_photo(image_path, caption=message)
                else:
                    self.send_message(message)
            except Exception as e:
                print(f"❌ Async alert send failed: {e}")

        # Start background thread
        thread = Thread(target=_send, daemon=True)
        thread.start()

    def send_danger_alert(self, frame, current_count, threshold, location="Unknown"):
        """
        Send danger alert with screenshot to Telegram (async).

        Args:
            frame: Current video frame (numpy array)
            current_count: Current number of people detected
            threshold: Danger threshold
            location: Location name

        Returns:
            bool: True if alert queued successfully
        """
        if not self.enabled:
            return False

        # Check cooldown to avoid spam
        current_time = time.time()
        if current_time - self.last_alert_time < self.alert_cooldown:
            return False  # Skip alert (too soon)

        try:
            # Clone frame to avoid race conditions
            frame_copy = frame.copy()

            # Prepare data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"danger_alert_{timestamp}.jpg"
            filepath = self.screenshots_dir / filename
            alert_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            message = (
                f"🚨 <b>DANGER ALERT</b> 🚨\n\n"
                f"📍 <b>Location:</b> {location}\n"
                f"👥 <b>People Count:</b> {current_count}\n"
                f"⚠️ <b>Threshold:</b> {threshold}\n"
                f"⏰ <b>Time:</b> {alert_time}\n\n"
                f"❗ Current occupancy exceeds safety limit!"
            )

            # Save and send in background thread (non-blocking)
            def _save_and_send_danger():
                try:
                    cv2.imwrite(str(filepath), frame_copy)
                    success = self.send_photo(str(filepath), caption=message)
                    if success:
                        print(f"✅ Danger alert sent successfully to Telegram!")
                except Exception as e:
                    print(f"❌ Failed to send danger alert in background: {e}")

            from threading import Thread
            Thread(target=_save_and_send_danger, daemon=True).start()
            print(f"[TELEGRAM] 📤 Danger alert queued (async) for {location}")

            # Update cooldown timestamp immediately (even though send is async)
            self.last_alert_time = current_time
            return True

        except Exception as e:
            print(f"❌ Failed to queue danger alert: {e}")
            return False

    def test_connection(self):
        """
        Test Telegram bot connection.

        Returns:
            bool: True if connection successful
        """
        if not self.enabled:
            print("❌ Telegram notifier is not enabled")
            return False

        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/getMe"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                bot_info = response.json()
                bot_name = bot_info.get('result', {}).get('username', 'Unknown')
                print(f"✅ Telegram bot connected: @{bot_name}")

                # Send test message
                test_msg = f"✅ <b>Crowd Detection System Connected</b>\n\nBot is ready to send danger alerts!"
                self.send_message(test_msg)

                return True
            else:
                print(f"❌ Telegram bot connection failed: {response.status_code}")
                return False

        except Exception as e:
            print(f"❌ Telegram connection error: {e}")
            return False


# Load from environment variables
def create_telegram_notifier():
    """
    Create TelegramNotifier instance from environment variables.

    Returns:
        TelegramNotifier: Configured notifier instance
    """
    import os
    from dotenv import load_dotenv

    # Load .env file
    load_dotenv()

    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')

    return TelegramNotifier(bot_token=bot_token, chat_id=chat_id)
