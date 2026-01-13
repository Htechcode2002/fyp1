from PySide6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, QFrame,
                               QScrollArea, QFormLayout, QLineEdit, QCheckBox, QHBoxLayout,
                               QSizePolicy, QRadioButton, QButtonGroup, QFileDialog, QGraphicsDropShadowEffect, QComboBox, QSpinBox)
from PySide6.QtGui import (QAction, QIcon, QPixmap, QDesktopServices, QIntValidator, 
                           QColor, QCursor, QPainter, QLinearGradient, QBrush, 
                           QPen, QFont)
from PySide6.QtCore import Qt, QSize, QUrl, QPropertyAnimation, QEasingCurve, QPoint, QTimer
from src.core.config_manager import ConfigManager
import os
import uuid # For generating unique IDs

class CollapsibleBox(QWidget):
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        
        self.toggle_button = QPushButton(title)
        self.toggle_button.setCursor(Qt.PointingHandCursor)
        self.toggle_button.setStyleSheet("""
            QPushButton {
                text-align: left; 
                background-color: white; 
                border: none;
                border-radius: 8px;
                padding: 15px 20px;
                font-weight: bold;
                font-size: 15px;
                color: #1e293b;
            }
            QPushButton:hover {
                background-color: #f1f5f9;
            }
            QPushButton:checked {
                border-bottom-left-radius: 0;
                border-bottom-right-radius: 0;
            }
        """)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(False)
        self.toggle_button.clicked.connect(self.on_pressed)

        # Add Shadow
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 20))
        shadow.setOffset(0, 2)
        self.toggle_button.setGraphicsEffect(shadow)

        self.content_area = QFrame()
        self.content_area.setStyleSheet("""
            QFrame {
                background-color: white;
                border-bottom-left-radius: 8px;
                border-bottom-right-radius: 8px;
            }
        """)
        self.content_area.setMaximumHeight(0)
        self.content_area.setMinimumHeight(0)
        self.content_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.content_layout = QVBoxLayout(self.content_area)
        self.content_layout.setContentsMargins(25, 10, 25, 25)
        self.content_layout.setSpacing(15)
        
        # Animation
        self.anim = QPropertyAnimation(self.content_area, b"maximumHeight")
        self.anim.setDuration(300)
        self.anim.setEasingCurve(QEasingCurve.InOutQuad)
        self.anim.finished.connect(self.on_anim_finished)

        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 5, 0, 5) # Spacing between boxes
        layout.addWidget(self.toggle_button)
        layout.addWidget(self.content_area)

    def on_pressed(self):
        checked = self.toggle_button.isChecked()
        
        if checked:
            content_height = self.content_layout.sizeHint().height()
            self.anim.setStartValue(0)
            self.anim.setEndValue(content_height)
            self.anim.start()
        else:
            current_height = self.content_area.height()
            self.anim.setStartValue(current_height)
            self.anim.setEndValue(0)
            self.anim.start()

    def on_anim_finished(self):
        if self.toggle_button.isChecked():
            # Remove limit so it can grow if content is added
             self.content_area.setMaximumHeight(16777215)

    def add_widget(self, widget):
        self.content_layout.addWidget(widget)


class VideoSourceCard(QFrame):
    def __init__(self, parent=None, source_data=None, onDelete=None):
        super().__init__(parent)
        self.setStyleSheet("""
            VideoSourceCard {
                background-color: #f8fafc;
                border-radius: 8px;
                border: 1px solid #e2e8f0;
            }
            QLabel { color: #475569; font-weight: 600; font-size: 13px; }
            QLineEdit {
                padding: 10px;
                border: 1px solid #e2e8f0;
                border-radius: 6px;
                color: #333;
                background-color: white;
            }
            QLineEdit:focus { border: 1px solid #3b82f6; }
            QRadioButton { color: #333; spacing: 8px; }
            QRadioButton::indicator { width: 14px; height: 14px; }
        """)
        
        self.source_data = source_data or {}
        self.onDelete = onDelete
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)
        
        # Header (Title + Location + Delete)
        header_layout = QHBoxLayout()
        header_layout.setSpacing(15)
        
        # ID Badge
        self.source_id = self.source_data.get("id", str(uuid.uuid4())[:8])
        self.lbl_id = QLabel(f"ID: {self.source_id}")
        self.lbl_id.setStyleSheet("""
            background-color: #f1f5f9; color: #64748b; 
            font-size: 11px; font-family: monospace;
            padding: 4px 8px; border-radius: 4px; border: 1px solid #e2e8f0;
        """)
        
        # Helper for common input style
        input_style = """
            QLineEdit {
                border: 1px solid #e2e8f0; border-radius: 6px; padding: 8px 12px;
                background-color: white; color: #1e293b; font-size: 13px;
            }
            QLineEdit:hover { border-color: #cbd5e1; }
            QLineEdit:focus { border-color: #3b82f6; background-color: #eff6ff; }
        """

        # Location Input (Takes main stage now)
        self.inp_location = QLineEdit(self.source_data.get("location", ""))
        self.inp_location.setPlaceholderText("📍 Location (e.g. Front Gate)")
        self.inp_location.setStyleSheet(input_style + "QLineEdit { font-weight: 600; font-size: 14px; }")
        
        # Delete Button
        btn_delete = QPushButton("🗑️") 
        btn_delete.setCursor(Qt.PointingHandCursor)
        btn_delete.setFixedSize(32, 32)
        btn_delete.setStyleSheet("""
            QPushButton {
                background-color: transparent; border-radius: 6px;
                border: 1px solid transparent;
            }
            QPushButton:hover {
                background-color: #fee2e2; border: 1px solid #fecaca;
            }
        """)
        btn_delete.setToolTip("Delete Source")
        if self.onDelete:
            btn_delete.clicked.connect(lambda: self.onDelete(self))
            
        header_layout.addWidget(self.lbl_id)
        header_layout.addWidget(self.inp_location, stretch=1)
        header_layout.addWidget(btn_delete)
        layout.addLayout(header_layout)
        
        # Horizontal Divider
        line = QFrame()
        line.setFixedHeight(1)
        line.setStyleSheet("background-color: #e2e8f0;")
        layout.addWidget(line)
        
        # Type Selection (Radio Buttons)
        self.type_group = QButtonGroup()
        self.rb_url = QRadioButton("Stream URL")
        self.rb_file = QRadioButton("Video File")
        self.rb_camera = QRadioButton("Webcam")
        self.rb_cctv = QRadioButton("CCTV/RTSP")
        self.rb_url.setCursor(Qt.PointingHandCursor)
        self.rb_file.setCursor(Qt.PointingHandCursor)
        self.rb_camera.setCursor(Qt.PointingHandCursor)
        self.rb_cctv.setCursor(Qt.PointingHandCursor)
        self.type_group.addButton(self.rb_url)
        self.type_group.addButton(self.rb_file)
        self.type_group.addButton(self.rb_camera)
        self.type_group.addButton(self.rb_cctv)

        self.rb_url.toggled.connect(self.update_input_mode)
        self.rb_file.toggled.connect(self.update_input_mode)
        self.rb_camera.toggled.connect(self.update_input_mode)
        self.rb_cctv.toggled.connect(self.update_input_mode)

        type_layout = QHBoxLayout()
        type_layout.addWidget(self.rb_url)
        type_layout.addWidget(self.rb_file)
        type_layout.addWidget(self.rb_camera)
        type_layout.addWidget(self.rb_cctv)
        type_layout.addStretch()
        layout.addLayout(type_layout)
        
        # Input Container
        self.inp_path = QLineEdit(self.source_data.get("path", ""))
        self.inp_path.setPlaceholderText("Paste URL (http/rtsp) or Select File")
        self.inp_path.setStyleSheet(input_style)

        self.btn_browse = QPushButton("Select File")
        self.btn_browse.setCursor(Qt.PointingHandCursor)
        self.btn_browse.setStyleSheet("""
            QPushButton {
                background-color: #f1f5f9; color: #334155; font-weight: 600;
                border-radius: 6px; padding: 8px 16px; border: 1px solid #cbd5e1;
            }
            QPushButton:hover { background-color: #e2e8f0; border-color: #94a3b8; }
        """)
        self.btn_browse.clicked.connect(self.browse_file)
        self.btn_browse.setVisible(False)

        # Camera Selection Dropdown
        self.camera_selector = QComboBox()
        self.camera_selector.setStyleSheet("""
            QComboBox {
                padding: 10px 15px;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                background-color: white; color: #1e293b;
                font-weight: 600;
            }
            QComboBox:hover { border-color: #cbd5e1; }
            QComboBox:focus { border: 2px solid #3b82f6; background-color: #f0f9ff; }
            QComboBox::drop-down { border: none; width: 30px; }
            QComboBox::down-arrow { image: none; border-left: 5px solid transparent; border-right: 5px solid transparent; border-top: 5px solid #64748b; margin-top: 2px; }
        """)
        self.camera_selector.addItem("✨ Auto Detect (Recommended)", "auto")
        self.camera_selector.addItem("📸 Camera 0 (System Default)", "0")
        self.camera_selector.addItem("📸 Camera 1", "1")
        self.camera_selector.addItem("📸 Camera 2", "2")
        self.camera_selector.addItem("📸 Camera 3", "3")
        self.camera_selector.setVisible(False)

        # Set saved camera index if available
        if self.source_data.get("type") == "camera" and "path" in self.source_data:
            saved_path = str(self.source_data["path"])
            index = self.camera_selector.findData(saved_path)
            if index >= 0:
                self.camera_selector.setCurrentIndex(index)
            else:
                self.camera_selector.setCurrentIndex(0) # Default to Auto
        else:
            self.camera_selector.setCurrentIndex(0) # Default to Auto

        # RTSP/CCTV Configuration Form
        self.rtsp_form = QFrame()
        self.rtsp_form.setStyleSheet("""
            QFrame {
                background-color: #eff6ff;
                border: 1px solid #bfdbfe;
                border-radius: 6px;
            }
        """)
        self.rtsp_form.setVisible(False)

        rtsp_layout = QVBoxLayout(self.rtsp_form)
        rtsp_layout.setContentsMargins(15, 15, 15, 15)
        rtsp_layout.setSpacing(10)

        # RTSP Form Title
        rtsp_title = QLabel("📹 CCTV Configuration")
        rtsp_title.setStyleSheet("color: #1e40af; font-weight: bold; font-size: 13px; background: transparent; border: none;")
        rtsp_layout.addWidget(rtsp_title)

        # Get saved RTSP config if available
        rtsp_config = self.source_data.get("rtsp_config", {})

        # IP Address
        ip_layout = QHBoxLayout()
        lbl_ip = QLabel("IP Address:")
        lbl_ip.setStyleSheet("color: #1e293b; font-weight: 600; min-width: 100px; background: transparent; border: none;")
        self.inp_rtsp_ip = QLineEdit(rtsp_config.get("ip", "192.168.0.79"))
        self.inp_rtsp_ip.setPlaceholderText("e.g. 192.168.0.79")
        self.inp_rtsp_ip.setStyleSheet(input_style)
        ip_layout.addWidget(lbl_ip)
        ip_layout.addWidget(self.inp_rtsp_ip)
        rtsp_layout.addLayout(ip_layout)

        # Username
        user_layout = QHBoxLayout()
        lbl_user = QLabel("Username:")
        lbl_user.setStyleSheet("color: #1e293b; font-weight: 600; min-width: 100px; background: transparent; border: none;")
        self.inp_rtsp_user = QLineEdit(rtsp_config.get("username", "admin"))
        self.inp_rtsp_user.setPlaceholderText("Username")
        self.inp_rtsp_user.setStyleSheet(input_style)
        user_layout.addWidget(lbl_user)
        user_layout.addWidget(self.inp_rtsp_user)
        rtsp_layout.addLayout(user_layout)

        # Password
        pass_layout = QHBoxLayout()
        lbl_pass = QLabel("Password:")
        lbl_pass.setStyleSheet("color: #1e293b; font-weight: 600; min-width: 100px; background: transparent; border: none;")
        self.inp_rtsp_pass = QLineEdit(rtsp_config.get("password", ""))
        self.inp_rtsp_pass.setPlaceholderText("Password")
        self.inp_rtsp_pass.setEchoMode(QLineEdit.Password)
        self.inp_rtsp_pass.setStyleSheet(input_style)
        pass_layout.addWidget(lbl_pass)
        pass_layout.addWidget(self.inp_rtsp_pass)
        rtsp_layout.addLayout(pass_layout)

        # Channel and Subtype (in one row)
        channel_layout = QHBoxLayout()

        lbl_channel = QLabel("Channel:")
        lbl_channel.setStyleSheet("color: #1e293b; font-weight: 600; background: transparent; border: none;")
        self.inp_rtsp_channel = QLineEdit(str(rtsp_config.get("channel", 1)))
        self.inp_rtsp_channel.setPlaceholderText("1")
        self.inp_rtsp_channel.setValidator(QIntValidator(1, 32))
        self.inp_rtsp_channel.setFixedWidth(60)
        self.inp_rtsp_channel.setStyleSheet(input_style)

        lbl_subtype = QLabel("Subtype:")
        lbl_subtype.setStyleSheet("color: #1e293b; font-weight: 600; margin-left: 15px; background: transparent; border: none;")
        self.combo_rtsp_subtype = QComboBox()
        self.combo_rtsp_subtype.addItem("Main Stream (High Quality)", 0)
        self.combo_rtsp_subtype.addItem("Sub Stream (Recommended)", 1)
        self.combo_rtsp_subtype.setCurrentIndex(rtsp_config.get("subtype", 1))
        self.combo_rtsp_subtype.setStyleSheet("""
            QComboBox {
                padding: 8px 12px;
                border: 1px solid #e2e8f0;
                border-radius: 6px;
                background-color: white;
                color: #1e293b;
            }
            QComboBox:focus { border: 1px solid #3b82f6; }
        """)

        channel_layout.addWidget(lbl_channel)
        channel_layout.addWidget(self.inp_rtsp_channel)
        channel_layout.addWidget(lbl_subtype)
        channel_layout.addWidget(self.combo_rtsp_subtype, 1)
        rtsp_layout.addLayout(channel_layout)

        # Generated RTSP URL Display (Read-only)
        self.lbl_rtsp_url = QLabel("RTSP URL: <i>Will be generated automatically</i>")
        self.lbl_rtsp_url.setStyleSheet("""
            color: #64748b;
            font-size: 11px;
            padding: 8px;
            background-color: #f1f5f9;
            border-radius: 4px;
            border: 1px dashed #cbd5e1;
        """)
        self.lbl_rtsp_url.setWordWrap(True)
        rtsp_layout.addWidget(self.lbl_rtsp_url)

        # Connect inputs to update URL preview
        self.inp_rtsp_ip.textChanged.connect(self.update_rtsp_url_preview)
        self.inp_rtsp_user.textChanged.connect(self.update_rtsp_url_preview)
        self.inp_rtsp_pass.textChanged.connect(self.update_rtsp_url_preview)
        self.inp_rtsp_channel.textChanged.connect(self.update_rtsp_url_preview)
        self.combo_rtsp_subtype.currentIndexChanged.connect(self.update_rtsp_url_preview)

        path_layout = QHBoxLayout()
        path_layout.addWidget(self.inp_path)
        path_layout.addWidget(self.btn_browse)
        path_layout.addWidget(self.camera_selector)

        layout.addLayout(path_layout)
        layout.addWidget(self.rtsp_form)  # Add RTSP form below path layout

        # Danger Threshold Input (Using LineEdit instead of SpinBox)
        danger_layout = QHBoxLayout()
        lbl_danger = QLabel("⚠️ Danger Threshold:")
        lbl_danger.setStyleSheet("color: #d97706; font-weight: 600; font-size: 13px;")

        self.inp_danger = QLineEdit(str(self.source_data.get("danger_threshold", 100)))
        self.inp_danger.setPlaceholderText("Count")
        self.inp_danger.setValidator(QIntValidator(0, 9999))
        self.inp_danger.setFixedWidth(100)
        self.inp_danger.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.inp_danger.setStyleSheet("""
            QLineEdit {
                border: 1px solid #fbbf24; border-radius: 6px; padding: 6px;
                background-color: #fffbeb; color: #92400e; font-weight: bold;
            }
            QLineEdit:focus { border: 1px solid #d97706; }
        """)

        lbl_people = QLabel("people max")
        lbl_people.setStyleSheet("color: #64748b; font-size: 12px;")

        danger_layout.addWidget(lbl_danger)
        danger_layout.addWidget(self.inp_danger)
        danger_layout.addWidget(lbl_people)
        danger_layout.addStretch()

        layout.addLayout(danger_layout)

        # Loitering Threshold Input
        loitering_layout = QHBoxLayout()
        lbl_loitering = QLabel("⏱️ Loitering Threshold:")
        lbl_loitering.setStyleSheet("color: #7c3aed; font-weight: 600; font-size: 13px;")

        self.inp_loitering = QLineEdit(str(self.source_data.get("loitering_threshold", 5)))
        self.inp_loitering.setPlaceholderText("Seconds")
        self.inp_loitering.setValidator(QIntValidator(1, 300))
        self.inp_loitering.setFixedWidth(100)
        self.inp_loitering.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.inp_loitering.setStyleSheet("""
            QLineEdit {
                border: 1px solid #a78bfa; border-radius: 6px; padding: 6px;
                background-color: #f5f3ff; color: #5b21b6; font-weight: bold;
            }
            QLineEdit:focus { border: 1px solid #7c3aed; }
        """)

        lbl_seconds = QLabel("seconds (dwell time)")
        lbl_seconds.setStyleSheet("color: #64748b; font-size: 12px;")

        loitering_layout.addWidget(lbl_loitering)
        loitering_layout.addWidget(self.inp_loitering)
        loitering_layout.addWidget(lbl_seconds)
        loitering_layout.addStretch()

        layout.addLayout(loitering_layout)

        # Fall Detection Threshold Input
        fall_layout = QHBoxLayout()
        lbl_fall = QLabel("🚨 Fall Detection Threshold:")
        lbl_fall.setStyleSheet("color: #dc2626; font-weight: 600; font-size: 13px;")

        self.inp_fall = QLineEdit(str(self.source_data.get("fall_threshold", 2)))
        self.inp_fall.setPlaceholderText("Seconds")
        self.inp_fall.setValidator(QIntValidator(1, 10))
        self.inp_fall.setFixedWidth(100)
        self.inp_fall.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.inp_fall.setStyleSheet("""
            QLineEdit {
                border: 1px solid #f87171; border-radius: 6px; padding: 6px;
                background-color: #fef2f2; color: #991b1b; font-weight: bold;
            }
            QLineEdit:focus { border: 1px solid #dc2626; }
        """)

        lbl_fall_seconds = QLabel("seconds (down time)")
        lbl_fall_seconds.setStyleSheet("color: #64748b; font-size: 12px;")

        fall_layout.addWidget(lbl_fall)
        fall_layout.addWidget(self.inp_fall)
        fall_layout.addWidget(lbl_fall_seconds)
        fall_layout.addStretch()

        layout.addLayout(fall_layout)

        # Save Button
        btn_save = QPushButton("Save Changes")
        btn_save.setCursor(Qt.PointingHandCursor)
        btn_save.setStyleSheet("""
            QPushButton {
                background-color: #3b82f6;
                color: white;
                font-weight: bold;
                border-radius: 6px;
                padding: 10px;
                border: none;
                margin-top: 5px;
            }
            QPushButton:hover { background-color: #2563eb; }
        """)
        btn_save.clicked.connect(self.save)
        layout.addWidget(btn_save)

        # Status message label (initially hidden)
        self.lbl_save_status = QLabel("")
        self.lbl_save_status.setAlignment(Qt.AlignCenter)
        self.lbl_save_status.setStyleSheet("""
            QLabel {
                padding: 10px;
                border-radius: 6px;
                font-weight: bold;
                margin-top: 5px;
            }
        """)
        self.lbl_save_status.hide()
        layout.addWidget(self.lbl_save_status)

        # Set initial state
        source_type = self.source_data.get("type", "url")
        if source_type == "file":
            self.rb_file.setChecked(True)
        elif source_type == "camera":
            self.rb_camera.setChecked(True)
        elif source_type == "cctv":
            self.rb_cctv.setChecked(True)
        else:
            self.rb_url.setChecked(True)
        self.update_input_mode()

    def update_input_mode(self):
        is_file = self.rb_file.isChecked()
        is_camera = self.rb_camera.isChecked()
        is_cctv = self.rb_cctv.isChecked()

        self.btn_browse.setVisible(is_file)
        self.inp_path.setVisible(not is_camera and not is_cctv)
        self.camera_selector.setVisible(is_camera)
        self.rtsp_form.setVisible(is_cctv)

        if is_cctv:
            # Update RTSP URL preview when switching to CCTV mode
            self.update_rtsp_url_preview()
        elif is_camera:
            # For camera, show dropdown selector
            pass
        elif is_file:
            self.inp_path.setPlaceholderText("No file selected...")
        else:
            self.inp_path.setPlaceholderText("Paste YouTube link or Stream URL here...")

    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.avi *.mkv *.mov)")
        if file_path:
            self.inp_path.setText(file_path)

    def update_rtsp_url_preview(self):
        """Generate and display RTSP URL based on current form inputs"""
        ip = self.inp_rtsp_ip.text().strip()
        username = self.inp_rtsp_user.text().strip()
        password = self.inp_rtsp_pass.text().strip()
        channel = self.inp_rtsp_channel.text().strip() or "1"
        subtype = self.combo_rtsp_subtype.currentData()

        if not ip:
            self.lbl_rtsp_url.setText("RTSP URL: <i>Please enter IP address</i>")
            return

        if not username or not password:
            self.lbl_rtsp_url.setText("RTSP URL: <i>Please enter username and password</i>")
            return

        # Construct RTSP URL
        rtsp_url = f"rtsp://{username}:{password}@{ip}:554/cam/realmonitor?channel={channel}&subtype={subtype}"

        # Display with password masked for security
        display_url = f"rtsp://{username}:****@{ip}:554/cam/realmonitor?channel={channel}&subtype={subtype}"
        self.lbl_rtsp_url.setText(f"<b>RTSP URL:</b> <code>{display_url}</code>")
        self.lbl_rtsp_url.setToolTip(f"Full URL (hover to see): {rtsp_url}")

    def get_rtsp_url(self):
        """Generate the actual RTSP URL with real password"""
        ip = self.inp_rtsp_ip.text().strip()
        username = self.inp_rtsp_user.text().strip()
        password = self.inp_rtsp_pass.text().strip()
        channel = self.inp_rtsp_channel.text().strip() or "1"
        subtype = self.combo_rtsp_subtype.currentData()

        if not ip or not username or not password:
            return ""

        return f"rtsp://{username}:{password}@{ip}:554/cam/realmonitor?channel={channel}&subtype={subtype}"

    def save(self):
        try:
            # Validate danger threshold
            val = int(self.inp_danger.text())
            if val < 0:
                raise ValueError("Danger threshold cannot be negative")
        except ValueError as e:
            # Show error message
            self.show_status_message(f"❌ Error: Invalid danger threshold", success=False)
            return

        try:
            # Validate loitering threshold
            loitering_val = int(self.inp_loitering.text())
            if loitering_val < 1:
                raise ValueError("Loitering threshold must be at least 1 second")
        except ValueError as e:
            # Show error message
            self.show_status_message(f"❌ Error: Invalid loitering threshold", success=False)
            return

        try:
            # Validate fall threshold
            fall_val = int(self.inp_fall.text())
            if fall_val < 1:
                raise ValueError("Fall threshold must be at least 1 second")
        except ValueError as e:
            # Show error message
            self.show_status_message(f"❌ Error: Invalid fall threshold", success=False)
            return

        try:
            self.source_data["id"] = self.source_id # Ensure ID is saved
            self.source_data["name"] = self.inp_location.text() or f"Source {self.source_id}" # Use location or default ID
            self.source_data["location"] = self.inp_location.text()
            self.source_data["danger_threshold"] = val
            self.source_data["loitering_threshold"] = loitering_val
            self.source_data["fall_threshold"] = fall_val

            type_str = "url"
            path_val = self.inp_path.text()

            if self.rb_camera.isChecked():
                type_str = "camera"
                # Save camera index as path
                path_val = str(self.camera_selector.currentData())
            elif self.rb_file.isChecked():
                type_str = "file"
            elif self.rb_cctv.isChecked():
                type_str = "cctv"
                # Generate and save RTSP URL
                path_val = self.get_rtsp_url()
                if not path_val:
                    self.show_status_message("❌ Error: Please fill in all RTSP fields", success=False)
                    return
                # Save RTSP config
                self.source_data["rtsp_config"] = {
                    "ip": self.inp_rtsp_ip.text().strip(),
                    "username": self.inp_rtsp_user.text().strip(),
                    "password": self.inp_rtsp_pass.text().strip(),
                    "channel": self.inp_rtsp_channel.text().strip() or "1",
                    "subtype": self.combo_rtsp_subtype.currentData()
                }

            self.source_data["type"] = type_str
            self.source_data["path"] = path_val

            # Animation for feedback
            anim = QPropertyAnimation(self, b"pos")
            anim.setDuration(100)
            anim.setStartValue(self.pos())
            anim.setEndValue(self.pos() + QPoint(0, 5))
            anim.setEasingCurve(QEasingCurve.Type.OutBounce)
            anim.start()

            # Persist to disk
            ConfigManager().save_config()
            print(f"Saved source {self.source_id}: {self.source_data}")

            # Show success message
            location_name = self.inp_location.text() or f"Source {self.source_id}"
            self.show_status_message(f"✅ Successfully saved: {location_name}", success=True)

        except Exception as e:
            # Show error message if something goes wrong
            self.show_status_message(f"❌ Failed to save: {str(e)}", success=False)
            print(f"Error saving config: {e}")

    def show_status_message(self, message, success=True):
        """Display a temporary status message to the user."""
        self.lbl_save_status.setText(message)

        if success:
            # Green background for success
            self.lbl_save_status.setStyleSheet("""
                QLabel {
                    background-color: #dcfce7;
                    color: #166534;
                    border: 1px solid #86efac;
                    padding: 10px;
                    border-radius: 6px;
                    font-weight: bold;
                    margin-top: 5px;
                }
            """)
        else:
            # Red background for error
            self.lbl_save_status.setStyleSheet("""
                QLabel {
                    background-color: #fee2e2;
                    color: #991b1b;
                    border: 1px solid #fca5a5;
                    padding: 10px;
                    border-radius: 6px;
                    font-weight: bold;
                    margin-top: 5px;
                }
            """)

        # Show the label
        self.lbl_save_status.show()

        # Auto-hide after 3 seconds
        QTimer.singleShot(3000, self.lbl_save_status.hide)

    def get_data(self):
        try:
            val = int(self.inp_danger.text())
        except ValueError:
            val = 100

        try:
            loitering_val = int(self.inp_loitering.text())
        except ValueError:
            loitering_val = 5

        try:
            fall_val = int(self.inp_fall.text())
        except ValueError:
            fall_val = 2

        type_str = "url"
        path_val = self.inp_path.text()
        rtsp_config = None

        if self.rb_camera.isChecked():
            type_str = "camera"
            path_val = str(self.camera_selector.currentData())
        elif self.rb_file.isChecked():
            type_str = "file"
        elif self.rb_cctv.isChecked():
            type_str = "cctv"
            # Generate RTSP URL and save config
            path_val = self.get_rtsp_url()
            rtsp_config = {
                "ip": self.inp_rtsp_ip.text().strip(),
                "username": self.inp_rtsp_user.text().strip(),
                "password": self.inp_rtsp_pass.text().strip(),
                "channel": self.inp_rtsp_channel.text().strip() or "1",
                "subtype": self.combo_rtsp_subtype.currentData()
            }

        data = {
            "id": self.source_id,
            "name": self.inp_location.text() or f"Source {self.source_id}",
            "type": type_str,
            "path": path_val,
            "location": self.inp_location.text(),
            "danger_threshold": val,
            "loitering_threshold": loitering_val,
            "fall_threshold": fall_val
        }

        # Add RTSP config if CCTV type
        if rtsp_config:
            data["rtsp_config"] = rtsp_config

        return data




class ConfigPage(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("background-color: #f8fafc;")
        self.cm = ConfigManager()
        
        # Main Scroll Area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollBar:vertical { width: 10px; }") # Simple scrollbar fix
        
        content = QWidget()
        content.setStyleSheet(".QWidget { background-color: transparent; }") # Transparent content wrapper
        layout = QVBoxLayout(content)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(20)
        
        # Header
        lbl_title = QLabel("Configuration")
        lbl_title.setStyleSheet("font-size: 26px; font-weight: 800; color: #1e293b; margin-bottom: 5px;")
        lbl_sub = QLabel("Manage your system settings and video sources")
        lbl_sub.setStyleSheet("font-size: 15px; color: #64748b; margin-bottom: 20px;")
        
        layout.addWidget(lbl_title)
        layout.addWidget(lbl_sub)
        
        # Sections
        self.create_section(layout, "YOLO Model Configuration", self.create_yolo_form())
        self.create_section(layout, "Video Sources Management", self.create_video_form())
        
        layout.addStretch()
        
        scroll.setWidget(content)
        
        # Outer layout
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(scroll)

    def create_section(self, parent_layout, title, content_widget):
        box = CollapsibleBox(title)
        box.add_widget(content_widget)
        parent_layout.addWidget(box)


    def create_yolo_form(self):
        widget = QWidget()
        form = QFormLayout(widget)
        form.setSpacing(15)
        self.inp_model = QLineEdit(self.cm.get("yolo", {}).get("model_path", "models/yolov8n.pt"))
        self.inp_model.setStyleSheet("padding: 10px; border: 1px solid #e2e8f0; border-radius: 6px;")
        
        btn_browse_model = QPushButton("Browse")
        btn_browse_model.setCursor(Qt.PointingHandCursor)
        btn_browse_model.setStyleSheet("padding: 10px; border: 1px solid #e2e8f0; border-radius: 6px; background-color: #f1f5f9;")
        btn_browse_model.clicked.connect(self.browse_model)

        row_layout = QHBoxLayout()
        row_layout.addWidget(self.inp_model)
        row_layout.addWidget(btn_browse_model)

        # Save button for this section
        btn_save_yolo = QPushButton("Save Model")
        btn_save_yolo.setCursor(Qt.PointingHandCursor)
        btn_save_yolo.setStyleSheet("padding: 10px; background-color: #3b82f6; color: white; border-radius: 6px; font-weight: bold; border: none;")
        btn_save_yolo.clicked.connect(self.save_yolo_config)

        form.addRow("Model Path:", row_layout)
        form.addRow("", btn_save_yolo)
        return widget

    def browse_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select YOLO Model", "", "Model Files (*.pt)")
        if file_path:
            # Try to make path relative to CWD if possible
            try:
                rel_path = os.path.relpath(file_path, os.getcwd())
                # If it's effectively in a different drive or way up, just use abs
                if ".." in rel_path and not rel_path.startswith("models"): 
                     self.inp_model.setText(file_path)
                else:
                     self.inp_model.setText(rel_path)
            except:
                self.inp_model.setText(file_path)

    def save_yolo_config(self):
        path = self.inp_model.text()
        self.cm.set("yolo", {"model_path": path})
        print(f"Saved YOLO config: {path}")

    def create_video_form(self):
        self.video_container = QWidget()
        self.video_layout = QVBoxLayout(self.video_container)
        self.video_layout.setSpacing(15)
        self.video_layout.setContentsMargins(0,0,0,0)
        
        sources = self.cm.get("video_sources", [])
        for src in sources:
            self.add_video_card(src)
            
        # Add New Button
        btn_add = QPushButton("+ Add New Source")
        btn_add.setCursor(Qt.PointingHandCursor)
        btn_add.setStyleSheet("""
            QPushButton {
                border: 2px dashed #e2e8f0;
                border-radius: 8px;
                padding: 15px;
                color: #64748b;
                background-color: transparent;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover { 
                background-color: #f8fafc; 
                border-color: #cbd5e1;
                color: #475569;
            }
        """)
        btn_add.clicked.connect(self.add_new_source)
        
        wrapper = QWidget()
        wrapper_layout = QVBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(0,0,0,0)
        wrapper_layout.setSpacing(15)
        wrapper_layout.addWidget(self.video_container)
        wrapper_layout.addWidget(btn_add)
        
        return wrapper


    def add_video_card(self, source_data):
        card = VideoSourceCard(source_data=source_data, onDelete=self.remove_video_card)
        self.video_layout.addWidget(card)

    def add_new_source(self):
        # Generate new ID with uniqueness check
        while True:
            new_id = str(uuid.uuid4())[:8]
            # Check if this ID already exists
            sources = self.cm.get("video_sources", [])
            existing_ids = [s.get("id") for s in sources]
            if new_id not in existing_ids:
                break
        
        # Consistent defaulting: Name uses Location if available, else Source {ID} (handled in UI save)
        new_data = {"id": new_id, "name": f"Source {new_id}", "location": "", "type": "url", "path": "", "danger_threshold": 100}
        sources = self.cm.get("video_sources", [])
        sources.append(new_data)
        self.cm.set("video_sources", sources) 
        self.add_video_card(new_data)

    def remove_video_card(self, card):
        source_id = card.source_id
        sources = self.cm.get("video_sources", [])
        
        # Find and remove by ID (much safer than dict matching)
        sources = [s for s in sources if s.get("id") != source_id]
        self.cm.set("video_sources", sources)
        
        self.video_layout.removeWidget(card)
        card.deleteLater()

