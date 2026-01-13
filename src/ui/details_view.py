import sys
import os
import time
import json
import warnings
import datetime
from PySide6.QtWidgets import (QDialog, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                               QPushButton, QFrame, QTableWidget, QTableWidgetItem,
                               QHeaderView, QSizePolicy, QStackedLayout, QMessageBox,
                               QScrollArea)
from PySide6.QtCore import Qt, QSize, QEvent
from PySide6.QtGui import QIcon, QPixmap, QCursor
from src.core.video import VideoThread
from src.ui.widgets import OverlayWidget

class VideoDetailDialog(QDialog):
    def __init__(self, parent=None, video_title="New Video Source", video_source=None, thread=None, main_window=None):
        super().__init__(parent)
        self.setWindowTitle(video_title)
        self.video_title = video_title
        self.resize(1100, 750)
        self.setStyleSheet("background-color: #f8fafc;")
        # Enable window maximize/minimize
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinMaxButtonsHint)

        self.source = video_source
        self.thread = thread
        self.is_own_thread = (thread is None) # If thread passed, we don't own it
        self.main_window = main_window  # Store reference to MainWindow
        
        self.source_size = None # QSize of original video
        self.display_size = None # QSize of UI widget
        self.off_x = 0
        self.off_y = 0
        self.display_img_size = None
        
        self.previous_counts = {} # Track previous counts for flash effect
        
        # Sync recording state with thread
        self.is_recording = thread.is_recording if thread else False
        self.recording_start_time = thread.recording_start_time if (thread and hasattr(thread, 'recording_start_time')) else None
        self.current_recording_path = thread.recording_output_path if (thread and hasattr(thread, 'recording_output_path')) else None
        
        # Initialize Detection States
        self.detection_active = True
        self.heatmap_active = False
        self.fall_detection_active = True
        self.face_analysis_active = True
        self.mask_detection_active = True
        self.display_mode = "dot" # Default to head dot
        self.tracking_active = True

        # Sync with actual thread states if available
        if thread:
            self.detection_active = thread.detection_enabled
            self.tracking_active = thread.tracking_enabled
            if hasattr(thread.detector, "heatmap_enabled"):
                self.heatmap_active = thread.detector.heatmap_enabled
            # Note: other AI toggles might need sync depending on thread implementation
        
        # Ensure recordings directory exists
        if not os.path.exists("recordings"):
            try:
                os.makedirs("recordings")
            except:
                pass



        # Main Layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(25, 25, 25, 25)
        self.main_layout.setSpacing(20)

        # --- Top Analytics Header ---
        header_panel = QFrame()
        header_panel.setFixedHeight(120)
        header_panel.setStyleSheet("background-color: white; border-radius: 12px; border: 1px solid #e2e8f0;")
        header_layout = QHBoxLayout(header_panel)
        header_layout.setContentsMargins(25, 10, 25, 10)
        
        # Title Group
        title_group = QVBoxLayout()
        lbl_title = QLabel(self.video_title)
        lbl_title.setStyleSheet("font-size: 22px; font-weight: 800; color: #1e293b; letter-spacing: -0.5px;")
        lbl_source = QLabel(f"Source: {self.source}")
        lbl_source.setStyleSheet("font-size: 13px; color: #64748b; font-weight: 600;")
        title_group.addWidget(lbl_title)
        title_group.addWidget(lbl_source)
        header_layout.addLayout(title_group)
        header_layout.addStretch()

        # Analytics Card Group
        self.stat_active = self._create_stat_card("ACTIVE NOW", "0", "👥", "#3b82f6")
        self.stat_left_total = self._create_stat_card("LEFT FLOW", "0", "⬅️", "#8b5cf6")
        self.stat_right_total = self._create_stat_card("RIGHT FLOW", "0", "➡️", "#ec4899")

        header_layout.addWidget(self.stat_active)
        header_layout.addWidget(self.stat_left_total)
        header_layout.addWidget(self.stat_right_total)

        self.main_layout.addWidget(header_panel)

        # Content Area (Video + Sidebar)
        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)

        # --- LEFT COLUMN: Video & Feed ---
        left_column = QWidget()
        left_layout = QVBoxLayout(left_column)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(15)

        # 1. Video Container
        self.video_container = QFrame()
        self.video_container.setStyleSheet("background-color: #0f172a; border-radius: 12px; border: 2px solid #1e293b;")
        self.video_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        container_layout = QVBoxLayout(self.video_container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(1, 1) # CRITICAL: Allow label to shrink
        self.video_label.setStyleSheet("background-color: transparent; border-radius: 10px;")
        self.video_label.installEventFilter(self)
        container_layout.addWidget(self.video_label)
        
        self.overlay_widget = OverlayWidget(self.video_label)
        self.overlay_widget.setAttribute(Qt.WA_TranslucentBackground)
        self.overlay_widget.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.overlay_widget.show()
        
        # Performance Badge (Overlay)
        self.lbl_perf = QLabel(self.video_label)
        self.lbl_perf.setStyleSheet("""
            background-color: rgba(15, 23, 42, 200);
            color: #3b82f6;
            font-weight: 800;
            font-size: 11px;
            padding: 6px 12px;
            border-radius: 6px;
            border: 1px solid rgba(59, 130, 246, 120);
        """)
        self.lbl_perf.setText("GPU INF: 0.0ms | TOTAL LATENCY: 0.0ms")
        self.lbl_perf.move(20, 20)
        self.lbl_perf.show()
        
        # 2. Events Table
        table_frame = QFrame()
        table_frame.setFixedHeight(220)
        table_frame.setStyleSheet("background-color: white; border-radius: 12px; border: 1px solid #e2e8f0;")
        table_layout = QVBoxLayout(table_frame)
        table_layout.setContentsMargins(20, 15, 20, 20)

        lbl_table_header = QLabel("🛰️ Real-time Event Log")
        lbl_table_header.setStyleSheet("font-weight: 800; font-size: 14px; color: #1e293b; text-transform: uppercase; letter-spacing: 0.5px;")
        
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Time", "Indicator", "Left", "Right", "Total Impact"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setFrameShape(QFrame.NoFrame)
        self.table.setStyleSheet("""
            QTableWidget {
                border: none;
                background: white;
                alternate-background-color: #f8fafc;
                selection-background-color: #eff6ff;
                selection-color: #1e293b;
            }
            QHeaderView::section {
                background-color: white;
                color: #64748b;
                font-weight: 800;
                font-size: 11px;
                border-bottom: 2px solid #f1f5f9;
                padding: 10px;
                text-transform: uppercase;
            }
            QTableWidget::item {
                padding: 10px;
                color: #334155;
                font-weight: 500;
                border-bottom: 1px solid #f1f5f9;
            }
        """)

        table_layout.addWidget(lbl_table_header)
        table_layout.addWidget(self.table)

        left_layout.addWidget(self.video_container, 1)
        left_layout.addWidget(table_frame, 0)
        
        # --- RIGHT COLUMN: Command Sidebar ---
        self.right_column_scroll = QScrollArea()
        self.right_column_scroll.setFixedWidth(320)
        self.right_column_scroll.setWidgetResizable(True)
        self.right_column_scroll.setFrameShape(QFrame.NoFrame)
        self.right_column_scroll.setStyleSheet("background-color: white; border-radius: 12px; border: 1px solid #e2e8f0;")
        
        sidebar_content = QWidget()
        sidebar_content.setStyleSheet("background: white;")
        sidebar_layout = QVBoxLayout(sidebar_content)
        sidebar_layout.setContentsMargins(20, 20, 20, 20)
        sidebar_layout.setSpacing(15)
        
        # Section Builder Helper
        def add_section(title, icon):
            label = QLabel(f"{icon}  {title}")
            label.setStyleSheet("color: #64748b; font-weight: 800; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; margin-top: 5px;")
            sidebar_layout.addWidget(label)

        # 1. COMMAND TOOLS
        add_section("Drawing Interface", "✍️")
        btn_draw_line = self._create_sidebar_btn("Define Count Line", "✏️")
        btn_draw_zone = self._create_sidebar_btn("Define Loiter Zone", "📐")
        btn_clear = self._create_sidebar_btn("Reset Interface", "🗑️")
        
        sidebar_layout.addWidget(btn_draw_line)
        sidebar_layout.addWidget(btn_draw_zone)
        sidebar_layout.addWidget(btn_clear)
        
        sidebar_layout.addSpacing(10)
        
        # 2. DETECTION SUITE
        add_section("Core System", "🛡️")
        self.btn_toggle_detection = self._create_toggle_btn("Primary Detection")
        sidebar_layout.addWidget(self.btn_toggle_detection)
        
        add_section("AI Intelligence", "🧠")
        self.btn_toggle_face_analysis = self._create_toggle_btn("Face/Age/Gender")
        self.btn_toggle_mask_detection = self._create_toggle_btn("Health Mask Scan")
        sidebar_layout.addWidget(self.btn_toggle_face_analysis)
        sidebar_layout.addWidget(self.btn_toggle_mask_detection)
        
        add_section("Safety Matrix", "⚡")
        self.btn_toggle_fall_detection = self._create_toggle_btn("Fall Emergency UI")
        sidebar_layout.addWidget(self.btn_toggle_fall_detection)
        
        add_section("Visualization", "🎨")
        self.btn_toggle_heatmap = self._create_toggle_btn("Dynamic Heatmap")
        self.btn_toggle_display_mode = self._create_toggle_btn("Head Tracking Dot") # Mode toggle
        sidebar_layout.addWidget(self.btn_toggle_heatmap)
        sidebar_layout.addWidget(self.btn_toggle_display_mode)
        
        sidebar_layout.addStretch()
        
        # 3. MISSION CONTROL
        add_section("Mission Control", "🎮")
        self.btn_record = QPushButton("⏺️ INITIALIZE RECORDING")
        self.btn_record.setFixedHeight(50)
        self.btn_record.setCursor(Qt.PointingHandCursor)
        sidebar_layout.addWidget(self.btn_record)

        self.right_column_scroll.setWidget(sidebar_content)

        content_layout.addWidget(left_column, stretch=1)
        content_layout.addWidget(self.right_column_scroll, stretch=0)
        
        self.main_layout.addLayout(content_layout)

        # Connect Logic
        btn_draw_line.clicked.connect(lambda: self.overlay_widget.set_mode("LINE"))
        btn_draw_zone.clicked.connect(lambda: self.overlay_widget.set_mode("ZONE"))
        btn_clear.clicked.connect(self.handle_clear_shapes)
        
        self.overlay_widget.line_drawn.connect(self.handle_line_drawn)
        self.overlay_widget.zone_drawn.connect(self.handle_zone_drawn)
        
        self.btn_toggle_detection.clicked.connect(self.toggle_detection)
        self.btn_toggle_heatmap.clicked.connect(self.toggle_heatmap)
        self.btn_toggle_fall_detection.clicked.connect(self.toggle_fall_detection)
        self.btn_toggle_face_analysis.clicked.connect(self.toggle_face_analysis)
        self.btn_toggle_mask_detection.clicked.connect(self.toggle_mask_detection)
        self.btn_toggle_display_mode.clicked.connect(self.toggle_display_mode)
        self.btn_record.clicked.connect(self.toggle_recording)

        self.update_all_states()
        self.start_video()

    def _create_sidebar_btn(self, text, icon):
        btn = QPushButton(f"  {icon}  {text}")
        btn.setFixedHeight(40)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setStyleSheet("""
            QPushButton {
                background-color: white;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 5px 15px;
                color: #475569;
                font-weight: 700;
                font-size: 11px;
                text-align: left;
            }
            QPushButton:hover {
                background-color: #f8fafc;
                border-color: #cbd5e1;
                color: #1e293b;
            }
            QPushButton:pressed {
                background-color: #f1f5f9;
            }
        """)
        return btn

    def _create_toggle_btn(self, text):
        btn = QPushButton(text)
        btn.setFixedHeight(40)
        btn.setCheckable(True)
        btn.setCursor(Qt.PointingHandCursor)
        return btn

    def update_all_states(self):
        self.update_detection_style()
        self.update_heatmap_style()
        self.update_fall_detection_style()
        self.update_face_analysis_style()
        self.update_mask_detection_style()
        self.update_display_mode_style()
        self.update_record_style()

    def update_detection_style(self):
        self._apply_toggle_style(self.btn_toggle_detection, self.detection_active, "Primary Detection")

    def _apply_toggle_style(self, btn, is_on, text):
        status = "ENABLED" if is_on else "DISABLED"
        btn.setText(f"{text}: {status}")
        if is_on:
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #f0fdf4;
                    color: #166534;
                    border: 1px solid #bbf7d0;
                    border-radius: 8px;
                    padding: 5px 15px;
                    font-weight: 800; font-size: 11px;
                }
                QPushButton:hover { background-color: #dcfce7; }
            """)
        else:
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #f8fafc;
                    color: #64748b;
                    border: 1px solid #e2e8f0;
                    border-radius: 8px;
                    padding: 5px 15px;
                    font-weight: 800; font-size: 11px;
                }
                QPushButton:hover { background-color: #f1f5f9; }
            """)

    def toggle_detection(self):
        self.detection_active = not self.detection_active
        self.update_detection_style()
        if self.thread:
            self.thread.set_detection(self.detection_active)

    def _create_stat_card(self, title, value, icon, color):
        """Create a premium styled stat card"""
        card = QFrame()
        card.setFixedWidth(160)
        card.setStyleSheet(f"""
            QFrame {{
                background-color: transparent;
                border: none;
            }}
        """)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(0, 5, 0, 5)
        layout.setSpacing(2)

        header = QHBoxLayout()
        lbl_icon = QLabel(icon)
        lbl_icon.setStyleSheet(f"font-size: 16px; margin-right: 5px;")
        lbl_title = QLabel(title)
        lbl_title.setStyleSheet(f"color: #64748b; font-size: 10px; font-weight: 800; letter-spacing: 0.5px;")
        header.addWidget(lbl_icon)
        header.addWidget(lbl_title)
        header.addStretch()

        lbl_value = QLabel(value)
        lbl_value.setStyleSheet(f"color: #1e293b; font-size: 32px; font-weight: 900; letter-spacing: -1px;")
        lbl_value.setObjectName("value")

        layout.addLayout(header)
        layout.addWidget(lbl_value)
        
        # Subtle bottom accent
        accent = QFrame()
        accent.setFixedHeight(3)
        accent.setFixedWidth(40)
        accent.setStyleSheet(f"background-color: {color}; border-radius: 1px;")
        layout.addWidget(accent)

        return card

    def start_video(self):
        # Case 1: Shared Thread (Already running)
        if self.thread is not None:
             # Connect signals
             # Connect signals safely - disconnect any existing first to avoid double calls
             with warnings.catch_warnings():
                 warnings.simplefilter("ignore", RuntimeWarning)
                 try:
                    self.thread.frame_signal.disconnect(self.set_frame)
                 except (RuntimeError, TypeError):
                    pass
                 try:
                    self.thread.stats_signal.disconnect(self.update_stats)
                 except (RuntimeError, TypeError):
                    pass
                 
             self.thread.frame_signal.connect(self.set_frame)
             self.thread.stats_signal.connect(self.update_stats)
             
             # Sync Toggles
             self.detection_active = self.thread.detection_enabled
             self.tracking_active = self.thread.tracking_enabled
             if hasattr(self.thread.detector, "heatmap_enabled"):
                 self.heatmap_active = self.thread.detector.heatmap_enabled
                 self.update_heatmap_style()
             else:
                 self.heatmap_active = False
                 self.update_heatmap_style()
             
             # Sync Recording State
             if hasattr(self.thread, "is_recording"):
                 self.is_recording = self.thread.is_recording
                 self.update_record_style()

        # Case 2: Own Thread (New Source)
        elif self.source is not None:
             # Persist=True for tracking
             self.thread = VideoThread(self.source, resolution=None, tracker="botsort.yaml")
             self.is_own_thread = True
             self.thread.frame_signal.connect(self.set_frame)
             self.thread.stats_signal.connect(self.update_stats) 
             
             self.thread.set_detection(self.detection_active)
             self.thread.set_tracking(self.tracking_active)
             
             self.thread.start()

    # Duplicate methods removed - consolidated below
            
    def set_frame(self, image):
        # Image is now QImage, conversion to QPixmap must happen in UI thread
        pixmap = QPixmap.fromImage(image)
        
        # Store original size for scaling
        self.source_size = pixmap.size()
        
        # Get the current container size to avoid feedback loops
        # Use a slightly smaller size than the actual label to be safe, 
        # or just ensure the pixmap doesn't trigger a resize.
        label_size = self.video_label.size()
        
        if label_size.width() <= 1 or label_size.height() <= 1:
            # First frame or hidden
            scaled_pixmap = pixmap.scaledToWidth(1280, Qt.SmoothTransformation)
        else:
            # Scale to fit while keeping aspect ratio
            scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
        self.video_label.setPixmap(scaled_pixmap)
        
        # Calculate offsets for drawing logic
        if not scaled_pixmap.isNull():
            self.display_img_size = scaled_pixmap.size()
            self.off_x = (label_size.width() - self.display_img_size.width()) // 2
            self.off_y = (label_size.height() - self.display_img_size.height()) // 2
        
        # If we have an overlay widget, update it too
        if hasattr(self, 'overlay_widget'):
            self.overlay_widget.setVideoSize(self.source_size)
        
        # Calculate Geometry of the actual image within the label
        w_label = label_size.width()
        h_label = label_size.height()
        w_pix = scaled_pixmap.width()
        h_pix = scaled_pixmap.height()
        
        # Center offset
        off_x = (w_label - w_pix) // 2
        off_y = (h_label - h_pix) // 2
        
        # Reposition OverlayWidget to match EXACTLY the video image
        # This completely solves the aspect ratio / black bar coordinate issue
        if hasattr(self, 'overlay_widget'):
            self.overlay_widget.setGeometry(off_x, off_y, w_pix, h_pix)
            self.overlay_widget.raise_()

    def handle_line_drawn(self, line):
        """
        Called when a new line is drawn. Parameter 'line' is normalized [(x1,y1), (x2,y2)].
        """
        if not self.thread or not self.source_size:
            return

        # Map Normalized -> Source Resolution
        # line is [(x1_norm, y1_norm), (x2_norm, y2_norm)]
        
        src_w = self.source_size.width()
        src_h = self.source_size.height()
        
        # Simplify: Calculate all lines from the overlay widget
        # The widget is now the Truth for Normalized coordinates
        all_lines = []
        for l in self.overlay_widget.lines:
            # l is ((nx1, ny1), (nx2, ny2))
            nx1, ny1 = l[0]
            nx2, ny2 = l[1]
            
            sx1 = int(nx1 * src_w)
            sy1 = int(ny1 * src_h)
            sx2 = int(nx2 * src_w)
            sy2 = int(ny2 * src_h)
            
            all_lines.append(((sx1, sy1), (sx2, sy2)))

        self.thread.set_lines(all_lines)

    def handle_zone_drawn(self, zone):
        """
        Called when a zone is drawn. Parameter 'zone' is normalized list of points.
        """
        if not self.thread or not self.source_size:
            return

        src_w = self.source_size.width()
        src_h = self.source_size.height()
        
        all_zones = []
        for z_points in self.overlay_widget.zones:
            # z_points is list of (nx, ny)
            zone_poly = []
            for (nx, ny) in z_points:
                sx = int(nx * src_w)
                sy = int(ny * src_h)
                zone_poly.append((sx, sy))
            all_zones.append(zone_poly)
            
        self.thread.set_zones(all_zones)

    def handle_clear_shapes(self):
        self.overlay_widget.clear_shapes()
        if self.thread:
            self.thread.set_lines([])
            self.thread.set_zones([])
            
    def update_stats(self, counts):
        # Update table with counts
        # counts is {line_index: {"left": x, "right": y, "total": z}, "_analytics": {...}}
        # print(f"DEBUG: update_stats called. Counts: {counts}") 

        # Update analytics cards if available
        if "_analytics" in counts:
            analytics = counts["_analytics"]

            # Update active pedestrians
            active_value = self.stat_active.findChild(QLabel, "value")
            if active_value:
                active_value.setText(str(analytics.get("active_pedestrians", 0)))

        # Calculate total left and right crossings across all lines
        total_left = 0
        total_right = 0
        
        # Check for Count Increases to Trigger Flash
        for key, data in counts.items():
            if key == "_analytics":
                continue
            
            # Key comes as string from signal now (for Qt compatibility), convert to int
            try:
                line_idx = int(key)
            except (ValueError, TypeError):
                continue
                
            current_total = data.get("total", 0)
            
            # Check previous count
            prev_total = self.previous_counts.get(line_idx, 0)
            
            if current_total > prev_total:
                # Trigger Flash!
                if hasattr(self, 'overlay_widget'):
                    self.overlay_widget.flash_line(line_idx)
            
            # Update history
            self.previous_counts[line_idx] = current_total

            total_left += data.get("left", 0)
            total_right += data.get("right", 0)
        
        # Update Performance Label
        perf = counts.get("_perf", {})
        if perf:
            inf = perf.get("inference", 0)
            lat = perf.get("latency", 0)
            fps = perf.get("fps", 0)
            self.lbl_perf.setText(f"🚀 GPU INF: {inf:.1f}ms | ⚡ TOTAL: {lat:.1f}ms | 📊 {fps:.1f} FPS")
            self.lbl_perf.adjustSize()
        # Update direction stat cards
        left_value = self.stat_left_total.findChild(QLabel, "value")
        if left_value:
            left_value.setText(str(total_left))

        right_value = self.stat_right_total.findChild(QLabel, "value")
        if right_value:
            right_value.setText(str(total_right))

        # Update table rows if needed
        # We only really need to update if counts changed (already throttled by signal)
        # But we also want to avoid flickering
        
        # Check if we need to rebuild the table (number of lines changed)
        line_count_items = [item for key, item in counts.items() if key != "_analytics"]
        if self.table.rowCount() != len(line_count_items):
            self.table.setRowCount(0)
            for i in range(len(line_count_items)):
                self.table.insertRow(i)
                for j in range(5):
                    self.table.setItem(i, j, QTableWidgetItem(""))

        import datetime
        import warnings
        now_str = datetime.datetime.now().strftime("%H:%M:%S")

        row = 0
        for key, data in counts.items():
            # Skip analytics metadata
            if key == "_analytics":
                continue
                
            try:
                line_idx = int(key)
            except (ValueError, TypeError):
                continue
            
            if row < self.table.rowCount():
                # Line Name (e.g., "Line 1")
                self.table.item(row, 0).setText(now_str)
                self.table.item(row, 1).setText(f"Line {line_idx+1}")

                # Counts
                left = data.get("left", 0)
                right = data.get("right", 0)
                total = data.get("total", 0)

                self.table.item(row, 2).setText(str(left))
                self.table.item(row, 3).setText(str(right))
                self.table.item(row, 4).setText(str(total))
                row += 1

    def eventFilter(self, source, event):
        if source is self.video_label and event.type() == QEvent.Resize:
            if hasattr(self, 'overlay_widget'):
                # Sync overlay size with label size
                self.overlay_widget.resize(event.size())
        return super().eventFilter(source, event)
    
    def paintEvent(self, event):
        super().paintEvent(event)

    def closeEvent(self, event):
        # We no longer stop recording on close, allowing background capture.
        
        # Disconnect signals to stop updates to this window
        if self.thread:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                try:
                    self.thread.frame_signal.disconnect(self.set_frame)
                except (RuntimeError, TypeError):
                    pass
                try:
                    self.thread.stats_signal.disconnect(self.update_stats)
                except (RuntimeError, TypeError):
                    pass

            # Only stop if we created it
            if self.is_own_thread:
                self.thread.stop()
                # Ensure the thread is finished before the dialog is destroyed
                self.thread.wait()
                
        event.accept()

    def update_heatmap_style(self):
        self._apply_toggle_style(self.btn_toggle_heatmap, self.heatmap_active, "Dynamic Heatmap")

    def toggle_heatmap(self):
        self.heatmap_active = not self.heatmap_active
        self.update_heatmap_style()
        if self.thread:
            self.thread.set_heatmap(self.heatmap_active)

    def update_fall_detection_style(self):
        self._apply_toggle_style(self.btn_toggle_fall_detection, self.fall_detection_active, "Fall Emergency UI")

    def toggle_fall_detection(self):
        self.fall_detection_active = not self.fall_detection_active
        self.update_fall_detection_style()
        if self.thread:
            self.thread.set_fall_detection(self.fall_detection_active)

    def toggle_face_analysis(self):
        """Toggle gender and age detection"""
        # Check if FaceAnalyzer is actually enabled/loaded
        if not self.face_analysis_active: # If we are trying to TURN IT ON
            if self.thread and hasattr(self.thread.detector, "face_analyzer"):
                fa = self.thread.detector.face_analyzer
                if not fa.enabled:
                    msg = fa.error_message if hasattr(fa, "error_message") and fa.error_message else "Face Analysis model failed to load."
                    QMessageBox.warning(self, "Face Analysis Unavailable", 
                                        f"Cannot enable face analysis:\n\n{msg}\n\n"
                                        "Please check your internet connection or download models manually.")
                    return

        self.face_analysis_active = not self.face_analysis_active
        self.update_face_analysis_style()
        if self.thread:
            self.thread.set_face_analysis_enabled(self.face_analysis_active)

    def update_face_analysis_style(self):
        self._apply_toggle_style(self.btn_toggle_face_analysis, self.face_analysis_active, "Face/Age/Gender")

    def toggle_mask_detection(self):
        """Toggle mask detection"""
        self.mask_detection_active = not self.mask_detection_active
        self.update_mask_detection_style()
        if self.thread:
            self.thread.set_mask_detection(self.mask_detection_active)

    def update_mask_detection_style(self):
        self._apply_toggle_style(self.btn_toggle_mask_detection, self.mask_detection_active, "Health Mask Scan")

    def toggle_display_mode(self):
        """Toggle between bounding box and head dot display"""
        if self.display_mode == "dot":
            self.display_mode = "box"
        else:
            self.display_mode = "dot"

        self.update_display_mode_style()
        if self.thread:
            self.thread.set_display_mode(self.display_mode)

    def update_display_mode_style(self):
        mode_text = "Tracking Dot" if self.display_mode == "dot" else "Bounding Box"
        self.btn_toggle_display_mode.setText(f"Display Mode: {mode_text}")
        self.btn_toggle_display_mode.setStyleSheet("""
            QPushButton {
                background-color: #eff6ff;
                color: #3b82f6;
                border: 1px solid #bfdbfe;
                border-radius: 8px;
                padding: 5px 15px;
                font-weight: 800; font-size: 11px;
            }
            QPushButton:hover { background-color: #dbeafe; }
        """)
    def toggle_recording(self):
        """Toggle video recording state"""
        if not self.thread:
            return

        if not self.is_recording:
            # Start Recording
            now = datetime.datetime.now()
            # self.recording_start_time will be set in the thread
            timestamp = now.strftime("%Y%m%d_%H%M%S")
            
            # Use Video Title (Location) as the folder name, sanitized
            clean_name = "".join([c for c in self.video_title if c.isalnum() or c in (' ', '_')]).strip().replace(' ', '_')
            
            # Create subfolder for this source
            source_dir = os.path.join("recordings", clean_name)
            if not os.path.exists(source_dir):
                os.makedirs(source_dir)
            
            # Initial temporary path
            self.current_recording_path = os.path.abspath(os.path.join(source_dir, f"{timestamp}_REC.mp4"))
            self.thread.start_recording(self.current_recording_path)
            self.is_recording = True
        else:
            # Stop Recording - Get final state from thread before it resets
            self.current_recording_path = self.thread.recording_output_path
            start_ts = self.thread.recording_start_time
            
            self.thread.stop_recording()
            self.is_recording = False
            
            # Rename the file to include end time for better searchability
            try:
                start_time = datetime.datetime.fromtimestamp(start_ts)
                end_time = datetime.datetime.now()
                
                date_str = start_time.strftime("%Y%m%d")
                start_hms = start_time.strftime("%H%M%S")
                end_hms = end_time.strftime("%H%M%S")
                
                old_path = self.current_recording_path
                dir_name = os.path.dirname(old_path)
                
                # New descriptive name: 20260112_153000__TO__153500.mp4
                new_filename = f"{date_str}_{start_hms}__TO__{end_hms}.mp4"
                new_path = os.path.join(dir_name, new_filename)
                
                # Small delay to ensure file handle is fully released by OS
                import time
                time.sleep(0.2)
                
                if os.path.exists(old_path):
                    if os.path.exists(new_path): # Handle collisions
                        new_path = new_path.replace(".mp4", f"_{int(time.time())}.mp4")
                    os.rename(old_path, new_path)
                    
                    # ALSO RENAME THUMBNAIL
                    old_thumb = old_path.replace(".mp4", "_thumb.jpg")
                    new_thumb = new_path.replace(".mp4", "_thumb.jpg")
                    if os.path.exists(old_thumb):
                        os.rename(old_thumb, new_thumb)
                        
                    print(f"[RECORD] 📝 Descriptive rename complete (including thumbnail): {new_filename}")
            except Exception as e:
                print(f"[RECORD] ❌ Failed to rename recording: {e}")
            
        self.update_record_style()

    def update_record_style(self):
        """Update recording button style"""
        if self.is_recording:
            self.btn_record.setText("🛑 TERMINATE RECORDING")
            self.btn_record.setStyleSheet("""
                QPushButton {
                    background-color: #ef4444;
                    color: white;
                    border: none;
                    border-radius: 12px;
                    font-weight: 900;
                    font-size: 12px;
                    letter-spacing: 1px;
                }
                QPushButton:hover {
                    background-color: #dc2626;
                }
            """)
        else:
            self.btn_record.setText("⏺️ INITIALIZE RECORDING")
            self.btn_record.setStyleSheet("""
                QPushButton {
                    background-color: #3b82f6;
                    color: white;
                    border: none;
                    border-radius: 12px;
                    font-weight: 900;
                    font-size: 12px;
                    letter-spacing: 1px;
                }
                QPushButton:hover {
                    background-color: #2563eb;
                }
            """)
