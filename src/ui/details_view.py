from PySide6.QtWidgets import (QDialog, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                               QPushButton, QFrame, QTableWidget, QTableWidgetItem,
                               QHeaderView, QSizePolicy, QStackedLayout, QMessageBox)
from PySide6.QtCore import Qt, QSize, QEvent
from PySide6.QtGui import QIcon, QPixmap, QCursor
from src.core.video import VideoThread
from src.ui.widgets import OverlayWidget

class VideoDetailDialog(QDialog):
    def __init__(self, parent=None, video_title="New Video Source", video_source=None, thread=None, main_window=None):
        super().__init__(parent)
        self.setWindowTitle(video_title)
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



        # Main Layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(20, 20, 20, 20)
        self.main_layout.setSpacing(15)

        # Content Area (Video + Sidebar)
        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)

        # --- LEFT COLUMN: Video & Detections ---
        left_column = QWidget()
        left_layout = QVBoxLayout(left_column)
        left_layout.setContentsMargins(0,0,0,0)
        left_layout.setSpacing(15)

        # 1. Video Container with Overlay
        self.video_container = QFrame()
        self.video_container.setMinimumHeight(450)
        self.video_container.setStyleSheet("background-color: black; border-radius: 8px;")
        self.video_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Use Standard Layout for frame
        container_layout = QVBoxLayout(self.video_container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        
        # Layer 0: Video Label
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; border-radius: 8px;")
        self.video_label.setScaledContents(False) 
        self.video_label.installEventFilter(self) # Catch resize events
        
        container_layout.addWidget(self.video_label)
        
        # Layer 1: Controls Overlay (Transparent Widget)
        # Layer 1: Drawing Overlay (transparent, catches clicks when in mode)
        # Overlay Logic: Manual Geometry
        # Make OverlayWidget a CHILD of video_label so it sits on top.
        # We will resize it manually in eventFilter.
        self.overlay_widget = OverlayWidget(self.video_label)
        self.overlay_widget.setAttribute(Qt.WA_TranslucentBackground)
        self.overlay_widget.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        # self.overlay_widget.installEventFilter(self) # We filter video_label instead
        
        # Ensure overlay shows
        self.overlay_widget.show()
        
        # 2. Analytics Summary Panel
        analytics_frame = QFrame()
        analytics_frame.setStyleSheet("background-color: white; border-radius: 8px; border: 1px solid #e0e0e0;")
        analytics_layout = QVBoxLayout(analytics_frame)

        lbl_analytics_title = QLabel("Pedestrian Analytics")
        lbl_analytics_title.setStyleSheet("font-weight: bold; font-size: 14px; color: #333; margin-bottom: 10px;")

        # Analytics Grid
        analytics_grid = QHBoxLayout()

        # Create stat cards
        self.stat_active = self._create_stat_card("Active Now", "0")
        self.stat_left_total = self._create_stat_card("Left Direction", "0")
        self.stat_right_total = self._create_stat_card("Right Direction", "0")

        analytics_grid.addWidget(self.stat_active)
        analytics_grid.addWidget(self.stat_left_total)
        analytics_grid.addWidget(self.stat_right_total)

        analytics_layout.addWidget(lbl_analytics_title)
        analytics_layout.addLayout(analytics_grid)

        # 3. Recent Detections Table
        self.table_frame = QFrame()
        self.table_frame.setStyleSheet("background-color: white; border-radius: 8px; border: 1px solid #e0e0e0;")
        table_layout = QVBoxLayout(self.table_frame)

        lbl_table_title = QLabel("Line Crossing Events")
        lbl_table_title.setStyleSheet("font-weight: bold; font-size: 14px; color: #333;")

        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Time", "Line", "Left", "Right", "Total"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setFrameShape(QFrame.NoFrame)
        self.table.setStyleSheet("""
            QTableWidget {
                border: none;
                gridline-color: #f1f5f9;
            }
            QHeaderView::section {
                background-color: white;
                color: #64748b;
                font-weight: bold;
                border-bottom: 1px solid #f1f5f9;
                padding: 8px;
            }
            QTableWidget::item {
                padding: 8px;
                color: #333;
            }
        """)

        table_layout.addWidget(lbl_table_title)
        table_layout.addWidget(self.table)

        left_layout.addWidget(analytics_frame, 0)

        self.video_container.setMinimumHeight(600) # Increased from 450
        left_layout.addWidget(self.video_container, 1) # Give video all extra vertical space
        left_layout.addWidget(self.table_frame, 0)
        
        # --- RIGHT COLUMN: Sidebar ---
        self.right_column = QFrame()
        self.right_column.setFixedWidth(300)
        self.right_column.setStyleSheet("background-color: white; border-radius: 8px; border: 1px solid #e0e0e0;")
        right_layout = QVBoxLayout(self.right_column)
        right_layout.setAlignment(Qt.AlignTop)
        
        # Collapse Header
        settings_header = QHBoxLayout()
        lbl_settings = QLabel("Visualization & Detection")
        lbl_settings.setStyleSheet("font-weight: bold; font-size: 14px; color: #333;")
        btn_collapse = QPushButton("v") 
        btn_collapse.setFixedSize(20, 20)
        btn_collapse.setCursor(Qt.PointingHandCursor)
        btn_collapse.setStyleSheet("border: none; color: #64748b;")
        
        settings_header.addWidget(btn_collapse)
        
        # --- Sidebar Content ---
        
        # 1. Drawing Controls Group
        lbl_drawing = QLabel("Drawing Controls")
        lbl_drawing.setStyleSheet("font-weight: bold; color: #475569; margin-top: 10px;")
        
        btn_draw_line = QPushButton(" Draw Line (Count)")
        btn_draw_zone = QPushButton(" Draw Zone (Loiter)")
        btn_clear = QPushButton(" Clear All")
        
        for btn in [btn_draw_line, btn_draw_zone, btn_clear]:
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: white;
                    border: 1px solid #cbd5e1;
                    border-radius: 6px;
                    padding: 8px;
                    color: #334155;
                    font-weight: 600;
                    text-align: left;
                }
                QPushButton:hover {
                    background-color: #f1f5f9;
                    border-color: #94a3b8;
                }
            """)
            
        # Connect Sidebar Buttons
        btn_draw_line.clicked.connect(lambda: self.overlay_widget.set_mode("LINE"))
        btn_draw_zone.clicked.connect(lambda: self.overlay_widget.set_mode("ZONE"))
        btn_clear.clicked.connect(self.handle_clear_shapes)

        # Connect Drawing Signals (Re-add here since removed from overlay block)
        self.overlay_widget.line_drawn.connect(self.handle_line_drawn)
        self.overlay_widget.zone_drawn.connect(self.handle_zone_drawn)
        
        # 2. Controls & Toggles
        
        # 2. Controls & Toggles
        
        # --- Toggle Detection (Boxes) ---
        btn_toggle_detection = QPushButton("✅ Detection is ON")
        btn_toggle_detection.setCheckable(True)
        btn_toggle_detection.setChecked(True)
        btn_toggle_detection.setCursor(Qt.PointingHandCursor)
        self.detection_active = True
        
        def update_detection_style():
            if self.detection_active:
                btn_toggle_detection.setText("✅ Detection is ON")
                btn_toggle_detection.setStyleSheet("""
                    QPushButton {
                        background-color: #10b981;
                        color: white;
                        border: none;
                        border-radius: 6px;
                        padding: 10px;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        background-color: #059669;
                    }
                """)
            else:
                btn_toggle_detection.setText("🛑 Detection is OFF")
                btn_toggle_detection.setStyleSheet("""
                    QPushButton {
                        background-color: #6b7280;
                        color: white;
                        border: none;
                        border-radius: 6px;
                        padding: 10px;
                        font-weight: bold;
                    }
                    QPushButton:hover {
                        background-color: #4b5563;
                    }
                """)

        def toggle_detection():
            self.detection_active = not self.detection_active
            update_detection_style()
            if self.thread:
                self.thread.set_detection(self.detection_active)
        
        btn_toggle_detection.clicked.connect(toggle_detection)
        update_detection_style() # Set initial style without flipping

        # --- Toggle Heatmap (New) ---
        self.btn_toggle_heatmap = QPushButton("🔥 Heatmap is OFF")
        self.btn_toggle_heatmap.setCheckable(True)
        self.btn_toggle_heatmap.setChecked(False)
        self.btn_toggle_heatmap.setCursor(Qt.PointingHandCursor)
        self.heatmap_active = False

        self.btn_toggle_heatmap.clicked.connect(self.toggle_heatmap)
        self.update_heatmap_style()

        # --- Toggle Fall Detection (New) ---
        self.btn_toggle_fall_detection = QPushButton("🚨 Fall Detection is ON")
        self.btn_toggle_fall_detection.setCheckable(True)
        self.btn_toggle_fall_detection.setChecked(True)
        self.btn_toggle_fall_detection.setCursor(Qt.PointingHandCursor)
        self.fall_detection_active = True

        lbl_fall_info = QLabel("Detects when people fall down")
        lbl_fall_info.setStyleSheet("color: #64748b; font-size: 11px; margin-bottom: 10px;")

        self.btn_toggle_fall_detection.clicked.connect(self.toggle_fall_detection)
        self.update_fall_detection_style()

        # Gender & Age Detection Toggle
        self.btn_toggle_face_analysis = QPushButton("👤 Gender & Age is ON")
        self.btn_toggle_face_analysis.setCursor(Qt.PointingHandCursor)
        self.face_analysis_active = True

        lbl_face_info = QLabel("Detects gender and age")
        lbl_face_info.setStyleSheet("color: #64748b; font-size: 11px; margin-bottom: 10px;")

        self.btn_toggle_face_analysis.clicked.connect(self.toggle_face_analysis)
        self.update_face_analysis_style()

        # Mask Detection Toggle
        self.btn_toggle_mask_detection = QPushButton("Mask Detection is ON")
        self.btn_toggle_mask_detection.setCursor(Qt.PointingHandCursor)
        self.mask_detection_active = True

        lbl_mask_info = QLabel("Detects face masks (with/without/incorrect)")
        lbl_mask_info.setStyleSheet("color: #64748b; font-size: 11px; margin-bottom: 10px;")

        self.btn_toggle_mask_detection.clicked.connect(self.toggle_mask_detection)
        self.update_mask_detection_style()

        # Display Mode Toggle (Box vs Dot)
        self.btn_toggle_display_mode = QPushButton("📦 Display: Head Dot")
        self.btn_toggle_display_mode.setCursor(Qt.PointingHandCursor)
        self.display_mode = "dot"  # "dot" or "box"

        lbl_display_info = QLabel("Switch between bounding box and head dot")
        lbl_display_info.setStyleSheet("color: #64748b; font-size: 11px; margin-bottom: 10px;")

        self.btn_toggle_display_mode.clicked.connect(self.toggle_display_mode)
        self.update_display_mode_style()

        right_layout.addLayout(settings_header)

        # ========== SECTION 1: DRAWING TOOLS ==========
        section1_label = QLabel("🎨 DRAWING TOOLS")
        section1_label.setStyleSheet("color: #1e293b; font-weight: bold; font-size: 12px; margin-top: 10px; margin-bottom: 5px;")
        right_layout.addWidget(section1_label)

        right_layout.addWidget(lbl_drawing)
        right_layout.addWidget(btn_draw_line)
        right_layout.addWidget(btn_draw_zone)
        right_layout.addWidget(btn_clear)

        # Divider
        divider1 = QFrame()
        divider1.setFrameShape(QFrame.HLine)
        divider1.setStyleSheet("background-color: #e2e8f0; margin: 15px 0;")
        right_layout.addWidget(divider1)

        # ========== SECTION 2: CORE DETECTION ==========
        section2_label = QLabel("🔍 CORE DETECTION")
        section2_label.setStyleSheet("color: #1e293b; font-weight: bold; font-size: 12px; margin-bottom: 5px;")
        right_layout.addWidget(section2_label)

        right_layout.addWidget(btn_toggle_detection)

        # Divider
        divider2 = QFrame()
        divider2.setFrameShape(QFrame.HLine)
        divider2.setStyleSheet("background-color: #e2e8f0; margin: 15px 0;")
        right_layout.addWidget(divider2)

        # ========== SECTION 3: AI ANALYSIS ==========
        section3_label = QLabel("🤖 AI ANALYSIS")
        section3_label.setStyleSheet("color: #1e293b; font-weight: bold; font-size: 12px; margin-bottom: 5px;")
        right_layout.addWidget(section3_label)

        right_layout.addWidget(self.btn_toggle_face_analysis)
        right_layout.addWidget(lbl_face_info)
        right_layout.addWidget(self.btn_toggle_mask_detection)
        right_layout.addWidget(lbl_mask_info)

        # Divider
        divider3 = QFrame()
        divider3.setFrameShape(QFrame.HLine)
        divider3.setStyleSheet("background-color: #e2e8f0; margin: 15px 0;")
        right_layout.addWidget(divider3)

        # ========== SECTION 4: SAFETY & ALERTS ==========
        section4_label = QLabel("⚠️ SAFETY & ALERTS")
        section4_label.setStyleSheet("color: #1e293b; font-weight: bold; font-size: 12px; margin-bottom: 5px;")
        right_layout.addWidget(section4_label)

        right_layout.addWidget(self.btn_toggle_fall_detection)
        right_layout.addWidget(lbl_fall_info)

        # Divider
        divider4 = QFrame()
        divider4.setFrameShape(QFrame.HLine)
        divider4.setStyleSheet("background-color: #e2e8f0; margin: 15px 0;")
        right_layout.addWidget(divider4)

        # ========== SECTION 5: VISUALIZATION ==========
        section5_label = QLabel("👁️ VISUALIZATION")
        section5_label.setStyleSheet("color: #1e293b; font-weight: bold; font-size: 12px; margin-bottom: 5px;")
        right_layout.addWidget(section5_label)

        right_layout.addWidget(self.btn_toggle_heatmap)
        right_layout.addWidget(self.btn_toggle_display_mode)
        right_layout.addWidget(lbl_display_info)

        right_layout.addStretch() 

        content_layout.addWidget(left_column, stretch=3)
        content_layout.addWidget(self.right_column, stretch=0)
        
        self.main_layout.addLayout(content_layout)
        
        # Start playback
        self.start_video()

    def _create_stat_card(self, title, value):
        """Create a styled stat card widget"""
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: #f8fafc;
                border-radius: 6px;
                padding: 10px;
            }
        """)
        layout = QVBoxLayout(card)
        layout.setSpacing(5)

        lbl_title = QLabel(title)
        lbl_title.setStyleSheet("color: #64748b; font-size: 11px; font-weight: 600;")

        lbl_value = QLabel(value)
        lbl_value.setStyleSheet("color: #1e293b; font-size: 20px; font-weight: bold;")
        lbl_value.setObjectName("value")

        layout.addWidget(lbl_title)
        layout.addWidget(lbl_value)

        return card

    def start_video(self):
        # Case 1: Shared Thread (Already running)
        if self.thread is not None:
             # Connect signals
             # Connect signals safely - disconnect any existing first to avoid double calls
             # Using Python's duck typing to check if connected is complex in PySide6
             # We just disconnect and ignore the warning/error if it wasn't connected
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
            
    def set_frame(self, pixmap):
        # Store original size for scaling
        self.source_size = pixmap.size()
        
        # Scale pixmap to fit label while maintaining aspect ratio
        label_size = self.video_label.size()
        if label_size.isEmpty():
            return
            
        scaled = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled)
        
        # Calculate Geometry of the actual image within the label
        w_label = label_size.width()
        h_label = label_size.height()
        w_pix = scaled.width()
        h_pix = scaled.height()
        
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
        
        # ... rest of function ...
        # Update direction stat cards
        left_value = self.stat_left_total.findChild(QLabel, "value")
        if left_value:
            left_value.setText(str(total_left))

        right_value = self.stat_right_total.findChild(QLabel, "value")
        if right_value:
            right_value.setText(str(total_right))

        self.table.setRowCount(0) # Clear existing rows

        import datetime
        now_str = datetime.datetime.now().strftime("%H:%M:%S")

        for key, data in counts.items():
            # Skip analytics metadata
            if key == "_analytics":
                continue
                
            try:
                line_idx = int(key)
            except (ValueError, TypeError):
                continue

            row = self.table.rowCount()
            self.table.insertRow(row)

            # Line Name (e.g., "Line 1")
            self.table.setItem(row, 0, QTableWidgetItem(now_str))
            self.table.setItem(row, 1, QTableWidgetItem(f"Line {line_idx+1}"))

            # Counts
            left = data.get("left", 0)
            right = data.get("right", 0)
            total = data.get("total", 0)

            self.table.setItem(row, 2, QTableWidgetItem(str(left)))
            self.table.setItem(row, 3, QTableWidgetItem(str(right)))
            self.table.setItem(row, 4, QTableWidgetItem(str(total)))

    def eventFilter(self, source, event):
        return super().eventFilter(source, event)
    
    def paintEvent(self, event):
        super().paintEvent(event)

    def closeEvent(self, event):
        # Disconnect signals to stop updates to this window
        if self.thread:
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
        if self.heatmap_active:
            self.btn_toggle_heatmap.setText("🔥 Heatmap is ON")
            self.btn_toggle_heatmap.setStyleSheet("""
                QPushButton {
                    background-color: #10b981;
                    color: white;
                    border: none;
                    border-radius: 6px;
                    padding: 10px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #059669;
                }
            """)
        else:
            self.btn_toggle_heatmap.setText("🔥 Heatmap is OFF")
            self.btn_toggle_heatmap.setStyleSheet("""
                QPushButton {
                    background-color: #6b7280;
                    color: white;
                    border: none;
                    border-radius: 6px;
                    padding: 10px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #4b5563;
                }
            """)

    def toggle_heatmap(self):
        self.heatmap_active = not self.heatmap_active
        self.update_heatmap_style()
        if self.thread:
            self.thread.set_heatmap(self.heatmap_active)

    def update_fall_detection_style(self):
        if self.fall_detection_active:
            self.btn_toggle_fall_detection.setText("🚨 Fall Detection is ON")
            self.btn_toggle_fall_detection.setStyleSheet("""
                QPushButton {
                    background-color: #10b981;
                    color: white;
                    border: none;
                    border-radius: 6px;
                    padding: 10px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #059669;
                }
            """)
        else:
            self.btn_toggle_fall_detection.setText("🚨 Fall Detection is OFF")
            self.btn_toggle_fall_detection.setStyleSheet("""
                QPushButton {
                    background-color: #6b7280;
                    color: white;
                    border: none;
                    border-radius: 6px;
                    padding: 10px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #4b5563;
                }
            """)

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
        """Update gender and age detection button style"""
        if self.face_analysis_active:
            self.btn_toggle_face_analysis.setText("👤 Gender & Age is ON")
            self.btn_toggle_face_analysis.setStyleSheet("""
                QPushButton {
                    background-color: #10b981;
                    color: white;
                    border: none;
                    border-radius: 6px;
                    padding: 10px;
                    font-weight: bold;
                    margin-bottom: 5px;
                }
                QPushButton:hover {
                    background-color: #059669;
                }
            """)
        else:
            self.btn_toggle_face_analysis.setText("👤 Gender & Age is OFF")
            self.btn_toggle_face_analysis.setStyleSheet("""
                QPushButton {
                    background-color: #6b7280;
                    color: white;
                    border: none;
                    border-radius: 6px;
                    padding: 10px;
                    font-weight: bold;
                    margin-bottom: 5px;
                }
                QPushButton:hover {
                    background-color: #4b5563;
                }
            """)

    def toggle_mask_detection(self):
        """Toggle mask detection"""
        self.mask_detection_active = not self.mask_detection_active
        self.update_mask_detection_style()
        if self.thread:
            self.thread.set_mask_detection(self.mask_detection_active)

    def update_mask_detection_style(self):
        """Update mask detection button style"""
        if self.mask_detection_active:
            self.btn_toggle_mask_detection.setText("Mask Detection is ON")
            self.btn_toggle_mask_detection.setStyleSheet("""
                QPushButton {
                    background-color: #10b981;
                    color: white;
                    border: none;
                    border-radius: 6px;
                    padding: 10px;
                    font-weight: bold;
                    margin-bottom: 5px;
                }
                QPushButton:hover {
                    background-color: #059669;
                }
            """)
        else:
            self.btn_toggle_mask_detection.setText("Mask Detection is OFF")
            self.btn_toggle_mask_detection.setStyleSheet("""
                QPushButton {
                    background-color: #6b7280;
                    color: white;
                    border: none;
                    border-radius: 6px;
                    padding: 10px;
                    font-weight: bold;
                    margin-bottom: 5px;
                }
                QPushButton:hover {
                    background-color: #4b5563;
                }
            """)

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
        """Update display mode button style"""
        # Display mode doesn't have ON/OFF states, both modes are "active"
        # Use green for current mode to match other buttons
        if self.display_mode == "box":
            self.btn_toggle_display_mode.setText("📦 Display: Bounding Box")
            self.btn_toggle_display_mode.setStyleSheet("""
                QPushButton {
                    background-color: #10b981;
                    color: white;
                    border: none;
                    border-radius: 6px;
                    padding: 10px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #059669;
                }
            """)
        else:  # "dot"
            self.btn_toggle_display_mode.setText("📦 Display: Head Dot")
            self.btn_toggle_display_mode.setStyleSheet("""
                QPushButton {
                    background-color: #10b981;
                    color: white;
                    border: none;
                    border-radius: 6px;
                    padding: 10px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #059669;
                }
            """)

