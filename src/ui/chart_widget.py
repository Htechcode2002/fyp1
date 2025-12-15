from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel
from PySide6.QtCore import Qt, QTimer, QPointF, QThread, Signal, QPoint, QRect
from PySide6.QtGui import QPainter, QPen, QColor, QFont, QLinearGradient, QBrush, QPainterPath, QCursor
from src.core.database import DatabaseManager
from datetime import datetime

class DataLoaderThread(QThread):
    data_loaded = Signal(list)

    def __init__(self, hours, interval):
        super().__init__()
        self.hours = hours
        self.interval = interval
        self.db = DatabaseManager()

    def run(self):
        try:
            # This runs in background thread
            raw_data = self.db.get_analytics_data(self.hours, self.interval)
            points = []
            for row in raw_data:
                # Assuming row keys match what MySQL returns
                dt = datetime.strptime(row['time_bucket'], '%Y-%m-%d %H:%M:%S')
                val = float(row['total_count'])
                points.append((dt, val))
            self.data_loaded.emit(points)
        except Exception as e:
            print(f"Chart Data Load Error: {e}")
            self.data_loaded.emit([])

class LineChartWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(350)
        
        # Layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(10)
        
        # Header Controls
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(10, 10, 10, 0)
        
        title = QLabel("People Traffic Trends")
        title.setStyleSheet("font-weight: bold; font-size: 14px; color: #334155;")
        
        self.combo_timeframe = QComboBox()
        self.combo_timeframe.addItems(["Last 1 Hour", "Last 24 Hours", "Last 7 Days"])
        self.combo_timeframe.currentIndexChanged.connect(self.request_refresh)
        self.combo_timeframe.setStyleSheet("""
            QComboBox {
                border: 1px solid #cbd5e1;
                border-radius: 4px;
                padding: 4px 8px;
                color: #475569;
                background: white;
            }
            QComboBox::drop-down { border: 0px; }
        """)

        header_layout.addWidget(title)
        header_layout.addStretch()
        header_layout.addWidget(self.combo_timeframe)
        
        self.layout.addLayout(header_layout)
        
        # Chart Canvas
        self.canvas = ChartCanvas(self)
        self.layout.addWidget(self.canvas)
        
        # Loading State
        self.loader = None
        
        # Auto refresh
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.request_refresh)
        self.timer.start(60000) # Every minute
        
        # Initial Load
        QTimer.singleShot(500, self.request_refresh)

    def request_refresh(self):
        if self.loader and self.loader.isRunning():
            return # Already loading

        index = self.combo_timeframe.currentIndex()
        hours = 1
        interval = 'minute'
        
        if index == 1: 
            hours = 24
            interval = 'hour'
        elif index == 2: 
            hours = 168 # 7 days
            interval = 'day'
        
        self.loader = DataLoaderThread(hours, interval)
        self.loader.data_loaded.connect(self.on_data_loaded)
        self.loader.start()
        
    def on_data_loaded(self, points):
        self.canvas.set_data(points, self.combo_timeframe.currentIndex())

class ChartCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.data = []
        self.timeframe_idx = 0
        self.setMouseTracking(True) # Enable mouse hover events
        self.hover_pos = None
        self.setStyleSheet("background-color: white; border-radius: 8px;")
        
    def set_data(self, data, timeframe_idx):
        self.data = data
        self.timeframe_idx = timeframe_idx
        self.update() 
        
    def mouseMoveEvent(self, event):
        self.hover_pos = event.pos()
        self.update()
        
    def leaveEvent(self, event):
        self.hover_pos = None
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w = self.width()
        h = self.height()
        
        # Background
        painter.fillRect(self.rect(), Qt.white)
        
        # Margins
        margin_left = 40
        margin_right = 20
        margin_top = 20
        margin_bottom = 40 # Increased for X-axis labels
        
        graph_w = w - margin_left - margin_right
        graph_h = h - margin_top - margin_bottom
        
        if not self.data:
            painter.setPen(QColor("#94a3b8"))
            painter.drawText(self.rect(), Qt.AlignCenter, "No Data Available or Loading...")
            return

        # Find Min/Max
        max_val = max(d[1] for d in self.data)
        if max_val == 0: max_val = 10
        # Add some headroom
        max_val = max_val * 1.1 
        
        start_time = self.data[0][0].timestamp()
        end_time = self.data[-1][0].timestamp()
        time_span = end_time - start_time
        if time_span == 0: time_span = 1
        
        # --- DRAW GRID & AXES ---
        painter.setFont(QFont("Arial", 8))
        
        # Y-Axis (Horizontal Grid)
        steps = 5
        painter.setPen(QPen(QColor("#f1f5f9"), 1))
        for i in range(steps + 1):
            y = margin_top + graph_h - (i / steps * graph_h)
            painter.drawLine(margin_left, int(y), w - margin_right, int(y))
            
            # Label
            val = int((i / steps) * max_val)
            painter.setPen(QColor("#64748b"))
            painter.drawText(0, int(y) - 5, margin_left - 5, 10, Qt.AlignRight, str(val))
            painter.setPen(QPen(QColor("#f1f5f9"), 1))

        # X-Axis Labels (Time)
        painter.setPen(QColor("#64748b"))
        label_count = 6
        for i in range(label_count):
            t_ratio = i / (label_count - 1)
            x = margin_left + (t_ratio * graph_w)
            
            # Interpolate time
            current_ts = start_time + (t_ratio * time_span)
            dt_obj = datetime.fromtimestamp(current_ts)
            
            # Format based on timeframe
            if self.timeframe_idx == 0: # 1H
                lbl = dt_obj.strftime("%H:%M")
            elif self.timeframe_idx == 1: # 24H
                lbl = dt_obj.strftime("%H:%M")
            else: # 7D
                lbl = dt_obj.strftime("%m-%d")
                
            # Draw Centered
            rect = QRect(int(x) - 20, h - margin_bottom + 5, 40, 20)
            painter.drawText(rect, Qt.AlignCenter, lbl)

        # --- PREPARE POINTS ---
        points_q = []
        for dt, count in self.data:
            t = dt.timestamp()
            x = margin_left + ((t - start_time) / time_span) * graph_w
            y = margin_top + graph_h - ((count / max_val) * graph_h)
            points_q.append(QPointF(x, y))
            
        if len(points_q) < 2:
            return

        # --- DRAW AREA & LINE ---
        path = QPainterPath()
        path.moveTo(points_q[0])
        for p in points_q[1:]:
            path.lineTo(p)
            
        # Linear Gradient Fill
        painter.setPen(Qt.NoPen)
        gradient = QLinearGradient(0, margin_top, 0, h - margin_bottom)
        gradient.setColorAt(0, QColor(59, 130, 246, 80)) 
        gradient.setColorAt(1, QColor(59, 130, 246, 5))
        
        area_path = QPainterPath(path)
        area_path.lineTo(points_q[-1].x(), h - margin_bottom)
        area_path.lineTo(points_q[0].x(), h - margin_bottom)
        area_path.closeSubpath()
        painter.setBrush(QBrush(gradient))
        painter.drawPath(area_path)
        
        # Stroke
        painter.setPen(QPen(QColor("#3b82f6"), 2))
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(path)
        
        # Draw Dots (only if not too many points)
        if len(self.data) < 50:
            painter.setBrush(QColor("white"))
            for p in points_q:
                painter.setPen(QPen(QColor("#3b82f6"), 2))
                painter.drawEllipse(p, 3, 3)

        # --- INTERACTIVE CROSSHAIR ---
        if self.hover_pos:
            mx = self.hover_pos.x()
            # Constrain to graph area
            if margin_left <= mx <= w - margin_right:
                # Draw Vertical Line
                painter.setPen(QPen(QColor("#94a3b8"), 1, Qt.DashLine))
                painter.drawLine(mx, margin_top, mx, h - margin_bottom)
                
                # Find closest point
                # Map mx back to time
                ratio = (mx - margin_left) / graph_w
                hover_ts = start_time + (ratio * time_span)
                
                # Simple closest search
                closest_pt = min(points_q, key=lambda p: abs(p.x() - mx))
                closest_idx = points_q.index(closest_pt)
                closest_data = self.data[closest_idx]
                
                # Highlight Point
                painter.setBrush(QColor("#ef4444"))
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(closest_pt, 5, 5)
                
                # Tooltip Box
                dt_str = closest_data[0].strftime("%Y-%m-%d %H:%M")
                val_str = f"{int(closest_data[1])} people"
                
                tooltip_text = f"{dt_str}\n{val_str}"
                
                # Draw Box
                bx = closest_pt.x() + 10
                by = closest_pt.y() - 40
                bw, bh = 120, 40
                
                # Stay within bounds
                if bx + bw > w: bx = closest_pt.x() - bw - 10
                
                painter.setBrush(QColor(30, 41, 59, 220)) # Dark slate
                painter.setPen(Qt.NoPen)
                painter.drawRoundedRect(bx, by, bw, bh, 5, 5)
                
                painter.setPen(QColor("white"))
                painter.drawText(QRect(int(bx), int(by), bw, bh), Qt.AlignCenter, tooltip_text)
