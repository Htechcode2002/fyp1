import os
import time
import datetime
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QPushButton, QFrame, QTableWidget, QTableWidgetItem, QHeaderView, QSizePolicy)
from PySide6.QtCore import Qt, Signal, QPoint, QRect, QTimer, QSize
from PySide6.QtGui import QPainter, QPen, QColor, QPolygon, QPixmap, QImage
from src.core.video import VideoThread

class VideoCard(QFrame):
    click_signal = Signal(str)

    def __init__(self, title="New Video Source", video_id=None):
        super().__init__()
        self.video_id = video_id
        self.setStyleSheet("""
            VideoCard {
                background-color: white;
                border-radius: 8px;
                border: 1px solid #e2e8f0;
            }
            QLabel#Title {
                font-weight: 700;
                font-size: 14px;
                color: #1e293b;
            }
            QLabel#Count {
                font-weight: 800;
                font-size: 42px;
                color: #0f172a;
            }
            QLabel#SubCount {
                color: #64748b;
                font-size: 10px;
                font-weight: 800;
                letter-spacing: 1px;
            }
            QPushButton#Details {
                background-color: #3b82f6; 
                color: white;
                border: none;
                border-radius: 4px;
                padding: 10px;
                font-weight: 700;
                font-size: 12px;
            }
            QPushButton#Details:hover {
                background-color: #2563eb;
            }
            QPushButton#Record {
                background-color: #f1f5f9;
                color: #475569;
                border: 1px solid #e2e8f0;
                border-radius: 4px;
                padding: 10px;
                font-weight: 600;
                font-size: 12px;
            }
            QPushButton#Record[recording="true"] {
                background-color: #ef4444;
                color: white;
                border: none;
            }
            QLabel#PerfBadge {
                background-color: rgba(15, 23, 42, 180);
                color: #3b82f6;
                font-weight: 800;
                font-size: 10px;
                padding: 4px 8px;
                border-radius: 4px;
                border: 1px solid rgba(59, 130, 246, 100);
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)

        # Video Placeholder
        self.video_label = QLabel()
        self.video_label.setFixedHeight(200)
        self.video_label.setStyleSheet("background-color: black; border-radius: 8px;")
        
        # Performance Badge (Overlay)
        self.lbl_perf = QLabel(self.video_label)
        self.lbl_perf.setObjectName("PerfBadge")
        self.lbl_perf.setText("GPU: 0ms")
        self.lbl_perf.move(10, 10)
        self.lbl_perf.show()
        
        # Live Badge (Overlay simulated by layout for now, or just a separate label)
        # For simplicity, just putting it above or inside. QStackLayout would be best for overlay.
        
        # Title
        self.lbl_title = QLabel(title)
        self.lbl_title.setObjectName("Title")

        # Stats Container
        stats_container = QFrame()
        stats_container.setStyleSheet("background-color: #f8fafc; border-radius: 8px; border: 1px solid #e2e8f0;")
        stats_layout = QVBoxLayout(stats_container)
        
        self.lbl_count = QLabel("0")
        self.lbl_count.setAlignment(Qt.AlignCenter)
        self.lbl_count.setObjectName("Count")
        
        lbl_subtitle = QLabel("PEOPLE DETECTED")
        lbl_subtitle.setAlignment(Qt.AlignCenter)
        lbl_subtitle.setObjectName("SubCount")
        
        stats_layout.addWidget(self.lbl_count)
        stats_layout.addWidget(lbl_subtitle)

        # Warning Pill removed - no longer needed

        # Buttons Container
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)

        # Details Button
        self.btn_details = QPushButton("View Details")
        self.btn_details.setObjectName("Details")
        self.btn_details.setCursor(Qt.PointingHandCursor)
        
        # Record Button
        self.btn_record = QPushButton("⏺️ Record")
        self.btn_record.setObjectName("Record")
        self.btn_record.setCursor(Qt.PointingHandCursor)
        self.btn_record.setProperty("recording", "false")
        self.btn_record.clicked.connect(self.toggle_recording)

        btn_layout.addWidget(self.btn_record, 1)
        btn_layout.addWidget(self.btn_details, 2)
        
        layout.addWidget(self.video_label)
        layout.addWidget(self.lbl_title)
        layout.addWidget(stats_container)
        layout.addLayout(btn_layout)

        self.thread = None
        self.source = None  # Store video source path
        self.is_recording = False

    def start_video(self, source, location=None, video_id=None, danger_threshold=100, loitering_threshold=5.0, fall_threshold=2.0):
        if self.thread and self.thread.isRunning():
            self.thread.stop()

        self.source = source  # Save the source path
        self.thread = VideoThread(source, location_name=location, video_id=video_id, danger_threshold=danger_threshold, loitering_threshold=loitering_threshold, fall_threshold=fall_threshold)
        self.thread.frame_signal.connect(self.set_frame)
        self.thread.stats_signal.connect(self.update_stats)
        self.thread.start()

    def stop_video(self):
        if self.thread:
            self.thread.stop()

    def set_frame(self, image):
        # Image is now QImage, must convert to QPixmap in GUI thread for safety
        pixmap = QPixmap.fromImage(image)
        # Scale to fit label height, keep aspect ratio
        scaled = pixmap.scaledToHeight(200, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled)

        # Sync recording UI state every frame (cheap check)
        if self.thread:
            thread_rec = self.thread.is_recording
            if thread_rec != self.is_recording:
                self.is_recording = thread_rec
                self._update_record_ui()
        
    def update_count(self, count):
        self.lbl_count.setText(str(count))

    def update_stats(self, data):
        # Extract active people count from analytics
        analytics = data.get("_analytics", {})
        active_count = analytics.get("active_pedestrians", 0)
        self.lbl_count.setText(str(active_count))

        # Update Performance Label
        perf = data.get("_perf", {})
        if perf:
            inf = perf.get("inference", 0)
            lat = perf.get("latency", 0)
            self.lbl_perf.setText(f"GPU: {inf:.1f}ms | Total: {lat:.1f}ms")
            self.lbl_perf.adjustSize()

    def _update_record_ui(self):
        """Update Record button appearance based on state"""
        if self.is_recording:
            self.btn_record.setText("⏹️ Stop")
            self.btn_record.setProperty("recording", "true")
        else:
            self.btn_record.setText("⏺️ Record")
            self.btn_record.setProperty("recording", "false")
        
        # Refresh style
        self.btn_record.style().unpolish(self.btn_record)
        self.btn_record.style().polish(self.btn_record)

    def toggle_recording(self):
        """Toggle recording on the thread"""
        if not self.thread:
            return

        if not self.is_recording:
            # Generate Path
            now = datetime.datetime.now()
            timestamp = now.strftime("%Y%m%d_%H%M%S")
            title = self.lbl_title.text()
            clean_name = "".join([c for c in title if c.isalnum() or c in (' ', '_')]).strip().replace(' ', '_')
            
            source_dir = os.path.abspath(os.path.join("recordings", clean_name))
            if not os.path.exists(source_dir):
                os.makedirs(source_dir)
            
            output_path = os.path.join(source_dir, f"{timestamp}_REC.mp4")
            self.thread.start_recording(output_path)
            self.is_recording = True
        else:
            # Stop
            dest_path = self.thread.recording_output_path
            start_ts = self.thread.recording_start_time
            
            self.thread.stop_recording()
            self.is_recording = False

            # Note: Final renaming (Start TO End) is handled by VideoThread itself in rotation logic,
            # but for manual stop, we should ideally handle it or let thread handle it.
            # Mirroring VideoDetailDialog logic for manual stop rename:
            if dest_path and os.path.exists(dest_path):
                try:
                    end_ts = time.time()
                    dt_start = datetime.datetime.fromtimestamp(start_ts)
                    dt_end = datetime.datetime.fromtimestamp(end_ts)
                    
                    new_name = f"{dt_start.strftime('%Y%m%d_%H%M%S')}__TO__{dt_end.strftime('%H%M%S')}.mp4"
                    new_path = os.path.join(os.path.dirname(dest_path), new_name)
                    
                    # Thumbnail rename
                    old_thumb = dest_path.replace(".mp4", "_thumb.jpg")
                    new_thumb = new_path.replace(".mp4", "_thumb.jpg")
                    
                    time.sleep(0.2)
                    if os.path.exists(dest_path):
                        os.rename(dest_path, new_path)
                        if os.path.exists(old_thumb):
                            os.rename(old_thumb, new_thumb)
                except Exception as e:
                    print(f"[CARD] Rename error: {e}")

        self._update_record_ui()


class AnalysisTable(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Header
        header_frame = QFrame()
        header_frame.setStyleSheet("""
            QFrame {
                background-color: #6366f1;
                border-top-left-radius: 12px;
                border-top-right-radius: 12px;
            }
            QLabel {
                color: white;
                font-weight: bold;
                font-size: 16px;
            }
            QLabel#Sub {
                color: #e0e7ff;
                font-size: 12px;
                font-weight: normal;
            }
        """)
        header_layout = QVBoxLayout(header_frame)
        lbl_main = QLabel("Real-time Location Analysis")
        lbl_sub = QLabel("Live monitoring data from all locations")
        lbl_sub.setObjectName("Sub")
        header_layout.addWidget(lbl_main)
        header_layout.addWidget(lbl_sub)
        
        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Location", "People Count", "Crowd Level", "Status", "Last Update"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #e0e0e0;
                border-bottom-left-radius: 12px;
                border-bottom-right-radius: 12px;
                gridline-color: #f1f5f9;
                background-color: white;
            }
            QHeaderView::section {
                background-color: white;
                padding: 10px;
                border-bottom: 1px solid #e0e0e0;
                font-weight: bold;
                color: #64748b;
            }
            QTableWidget::item {
                padding: 10px;
                border-bottom: 1px solid #f1f5f9;
            }
        """)
        
        layout.addWidget(header_frame)
        layout.addWidget(self.table)
        
    def add_row(self, location, count, level, status, time):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(location))
        
        # Center count and make it bold/blue
        item_count = QTableWidgetItem(str(count))
        item_count.setTextAlignment(Qt.AlignCenter)
        item_count.setForeground(Qt.darkBlue) # Simplified, better with font sizing
        self.table.setItem(row, 1, item_count)
        
        self.table.setItem(row, 2, QTableWidgetItem(level))
        self.table.setItem(row, 3, QTableWidgetItem(status))
        self.table.setItem(row, 4, QTableWidgetItem(time))

class OverlayWidget(QWidget):
    # Signals to notify when a shape is completed
    line_drawn = Signal(list) # [start_point, end_point]
    zone_drawn = Signal(list) # [p1, p2, p3, ...]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        self.mode = "NONE" # NONE, LINE, ZONE
        self.current_points = [] # In NORMALIZED coordinates
        self.temp_point = None # In PIXEL coordinates
        
        # Stored Shapes in NORMALIZED coordinates
        self.lines = [] # list of ((x1, y1), (x2, y2))
        self.zones = [] # list of [(x,y), (x,y), ...]
        
        # Flashing State
        self.flashing_lines = set() # Set of line indices currently flashing
        
        # Video Source Resolution (for coordinate mapping)
        self.video_size = QSize(1280, 720) # Default

    def set_mode(self, mode):
        self.mode = mode
        self.current_points = []
        self.temp_point = None
        self.setCursor(Qt.CrossCursor if mode != "NONE" else Qt.ArrowCursor)
        self.update()

    def setVideoSize(self, size):
        """Set the source video resolution for proper coordinate mapping."""
        if size and isinstance(size, QSize):
            self.video_size = size
            self.update()

    def add_line(self, line):
        self.lines.append(line)
        self.update()

    def add_zone(self, zone):
        self.zones.append(zone)
        self.update()

    def flash_line(self, index, duration=150):
        """Briefly highlight a line to indicate activity"""
        if 0 <= index < len(self.lines):
            self.flashing_lines.add(index)
            self.update()
            
            # Auto-remove after duration
            QTimer.singleShot(duration, lambda: self._stop_flash(index))

    def _stop_flash(self, index):
        if index in self.flashing_lines:
            self.flashing_lines.remove(index)
            self.update()

    def clear_shapes(self):
        self.lines = []
        self.zones = []
        self.current_points = []
        self.flashing_lines.clear()
        self.mode = "NONE"
        self.setCursor(Qt.ArrowCursor)
        self.update()

    def _to_normalized(self, pos):
        """Convert widget pixel position to normalized coordinates"""
        w = max(1, self.width())
        h = max(1, self.height())
        return (pos.x() / w, pos.y() / h)

    def _to_pixel(self, norm_pos):
        """Convert normalized coordinates to widget pixel position"""
        w = self.width()
        h = self.height()
        return QPoint(int(norm_pos[0] * w), int(norm_pos[1] * h))

    # mousePressEvent, mouseMoveEvent, mouseDoubleClickEvent, resizeEvent ... 
    # (These remain largely the same, but simplified for brevity in this replacement block if possible, 
    # but I must match the exact block to replace or provide full content if replacing a large chunk.
    # Logic requires me to replace from __init__ down to paintEvent to be safe and clean.)

    def mousePressEvent(self, event):
        if self.mode == "NONE":
            event.ignore()
            return

        if event.button() == Qt.LeftButton:
            norm_pos = self._to_normalized(event.pos())
            self.current_points.append(norm_pos)

            if self.mode == "LINE":
                if len(self.current_points) == 2:
                    self.lines.append(tuple(self.current_points))
                    self.line_drawn.emit(self.current_points)
                    self.current_points = []
                    self.temp_point = None
                    self.set_mode("NONE")

            elif self.mode == "ZONE":
                pass

            self.update()

        elif event.button() == Qt.RightButton:
            if self.mode == "ZONE" and len(self.current_points) >= 3:
                self.zones.append(self.current_points)
                self.zone_drawn.emit(self.current_points)
                self.current_points = []
                self.temp_point = None
                self.set_mode("NONE")
                self.update()
                event.accept()
            else:
                event.ignore()

    def mouseMoveEvent(self, event):
        if self.mode != "NONE":
            self.temp_point = event.pos() 
            self.update()
        else:
            self.temp_point = event.pos()
            event.ignore()

    def mouseDoubleClickEvent(self, event):
        if self.mode == "ZONE" and len(self.current_points) > 2:
            self.zones.append(self.current_points)
            self.zone_drawn.emit(self.current_points)
            self.current_points = []
            self.temp_point = None
            self.set_mode("NONE")
            self.update()
            event.accept()
        else:
            event.ignore()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 1. Draw Existing Lines 
        # Standard Pen
        pen_std = QPen(QColor(0, 255, 0), 3)
        # Flash Pen (Cyan, Thicker)
        pen_flash = QPen(QColor(0, 255, 255), 6)
        
        for i, (start_norm, end_norm) in enumerate(self.lines):
            p1 = self._to_pixel(start_norm)
            p2 = self._to_pixel(end_norm)
            
            if i in self.flashing_lines:
                painter.setPen(pen_flash)
            else:
                painter.setPen(pen_std)
                
            painter.drawLine(p1, p2)
            
        # 2. Draw Existing Zones
        pen_zone = QPen(QColor(255, 0, 0), 2)
        brush_zone = QColor(255, 0, 0, 50)
        painter.setPen(pen_zone)
        painter.setBrush(brush_zone)
        for points_norm in self.zones:
            if len(points_norm) > 1:
                pixel_points = [self._to_pixel(p) for p in points_norm]
                painter.drawPolygon(QPolygon(pixel_points))
        
        # 3. Draw In-Progress Drawing
        if self.mode == "LINE" and self.current_points:
            painter.setPen(QPen(QColor(0, 255, 0), 2, Qt.DashLine))
            painter.setBrush(Qt.NoBrush)
            
            start_pixel = self._to_pixel(self.current_points[0])
            end_pixel = self.temp_point if self.temp_point else start_pixel
            painter.drawLine(start_pixel, end_pixel)
            
        elif self.mode == "ZONE" and self.current_points:
            painter.setPen(QPen(QColor(255, 0, 0), 2, Qt.DashLine))
            painter.setBrush(Qt.NoBrush)
            
            pixel_points = [self._to_pixel(p) for p in self.current_points]
            
            for i in range(len(pixel_points) - 1):
                painter.drawLine(pixel_points[i], pixel_points[i+1])
            
            if self.temp_point and pixel_points:
                painter.drawLine(pixel_points[-1], self.temp_point)

