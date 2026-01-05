from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                               QPushButton, QFrame, QScrollArea, QStackedWidget, QSpacerItem, QSizePolicy)
from PySide6.QtCore import Qt
from src.ui.widgets import VideoCard
from src.ui.details_view import VideoDetailDialog
from src.ui.config_page import ConfigPage
from src.ui.data_view_page import DataViewPage
from src.core.config_manager import ConfigManager

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Initialize Config Manager to load defaults
        ConfigManager()

        self.setWindowTitle("Crowd Detection System")
        self.resize(1280, 850)
        self.setStyleSheet("background-color: #f8fafc;") # Light gray/blue bg

        # Store all video cards for cross-camera tracking
        self.video_cards = []

        # Main Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- Header ---
        header = QFrame()
        header.setFixedHeight(60)
        header.setStyleSheet("background-color: white; border-bottom: 1px solid #e0e0e0;")
        header_layout = QHBoxLayout(header)
        
        lbl_logo = QLabel("CD") # Placeholder for Icon
        lbl_logo.setStyleSheet("background-color: #3b82f6; color: white; border-radius: 4px; padding: 5px; font-weight: bold;")
        lbl_app_name = QLabel("Crowd Detection System")
        lbl_app_name.setStyleSheet("font-weight: bold; font-size: 16px; color: #333;")
        
        self.btn_dashboard = QPushButton("Dashboard")
        self.btn_data = QPushButton("Database")
        self.btn_config = QPushButton("Config")
        self.btn_dashboard.setCursor(Qt.PointingHandCursor)
        self.btn_data.setCursor(Qt.PointingHandCursor)
        self.btn_config.setCursor(Qt.PointingHandCursor)
        self.btn_dashboard.clicked.connect(lambda: self.switch_page(0))
        self.btn_data.clicked.connect(lambda: self.switch_page(1))
        self.btn_config.clicked.connect(lambda: self.switch_page(2))

        for btn in [self.btn_dashboard, self.btn_data, self.btn_config]:
            btn.setStyleSheet("border: none; color: #64748b; font-weight: bold;")

        header_layout.addWidget(lbl_logo)
        header_layout.addWidget(lbl_app_name)
        header_layout.addStretch()
        header_layout.addWidget(self.btn_dashboard)
        header_layout.addSpacing(10)
        header_layout.addWidget(self.btn_data)
        header_layout.addSpacing(10)
        header_layout.addWidget(self.btn_config)
        header_layout.setContentsMargins(20, 0, 20, 0)
        
        main_layout.addWidget(header)

        # --- Stacked Widget for Pages ---
        self.stack = QStackedWidget()

        # Page 0: Dashboard
        dashboard_page = self.create_dashboard_page()
        self.stack.addWidget(dashboard_page)

        # Page 1: Database Viewer
        data_page = DataViewPage()
        self.stack.addWidget(data_page)

        # Page 2: Config
        config_page = ConfigPage()
        self.stack.addWidget(config_page)

        main_layout.addWidget(self.stack)

        # Initial Refresh
        self.refresh_dashboard()

    def create_dashboard_page(self):
        """Creates the dashboard scroll area and content."""
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(40, 40, 40, 40)
        content_layout.setSpacing(20)

        # Dashboard elements (moved from __init__)
        title_container = QWidget()
        title_layout = QHBoxLayout(title_container)
        title_layout.setContentsMargins(0, 0, 0, 0)
        
        lbl_dash_title = QLabel("Video Monitoring Dashboard")
        lbl_dash_title.setStyleSheet("font-size: 24px; font-weight: bold; color: #1e293b;")
        
        
        # Refresh button removed - no longer needed
        
        title_layout.addWidget(lbl_dash_title)
        title_layout.addStretch()
        
        # Subtitle
        lbl_subtitle = QLabel("1 video source available")
        lbl_subtitle.setStyleSheet("color: #64748b;")

        content_layout.addWidget(title_container)
        content_layout.addWidget(lbl_subtitle)

        # Container for Video Cards
        self.cards_container = QWidget()
        self.cards_layout = QHBoxLayout(self.cards_container) # Use HBox for side-by-side or Grid
        self.cards_layout.setContentsMargins(0, 0, 0, 0)
        self.cards_layout.setAlignment(Qt.AlignLeft)
        
        content_layout.addWidget(self.cards_container)

        content_layout.addStretch()

        scroll_area.setWidget(content_widget)
        return scroll_area

    def refresh_dashboard(self):
        """Reload video sources from config and rebuild dashboard."""
        # Stop and join existing threads first
        for card in self.video_cards:
            card.stop_video()

        # Clear existing cards from layout
        while self.cards_layout.count():
            child = self.cards_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        # Clear video cards list
        self.video_cards = []

        cm = ConfigManager()
        sources = cm.get("video_sources", [])

        for src in sources:
            # Prioritize Location for the title if available, per user request
            name = src.get("name", "Unknown Source")
            location = src.get("location", "")
            video_id = src.get("id", "Unknown")
            display_title = location if location else name
            
            card = VideoCard(title=display_title)
            card.setFixedWidth(400)

            # Use closure to capture the specific card for this source
            card.btn_details.clicked.connect(lambda _, c=card: self.open_details(c))
            self.cards_layout.addWidget(card)

            # Add to video cards list
            self.video_cards.append(card)

            # Start Video
            path = src.get("path", "")
            danger_threshold = src.get("danger_threshold", 100)  # Get danger threshold from config
            loitering_threshold = src.get("loitering_threshold", 5.0)  # Get loitering threshold from config
            fall_threshold = src.get("fall_threshold", 2.0)  # Get fall threshold from config
            if path:
                # Basic check: if path is integer string (0, 1) treat as webcam index
                if path.isdigit():
                    card.start_video(int(path), location=location, video_id=video_id, danger_threshold=danger_threshold, loitering_threshold=loitering_threshold, fall_threshold=fall_threshold)
                else:
                    card.start_video(path, location=location, video_id=video_id, danger_threshold=danger_threshold, loitering_threshold=loitering_threshold, fall_threshold=fall_threshold)
            
    def switch_page(self, index):
        self.stack.setCurrentIndex(index)

        # Refresh if switching to dashboard
        if index == 0:
            self.refresh_dashboard()

        # Simple active state styling
        # Reset all buttons
        self.btn_dashboard.setStyleSheet("border: none; color: #64748b; font-weight: bold;")
        self.btn_data.setStyleSheet("border: none; color: #64748b; font-weight: bold;")
        self.btn_config.setStyleSheet("border: none; color: #64748b; font-weight: bold;")

        # Highlight active button
        if index == 0:
            self.btn_dashboard.setStyleSheet("border: none; color: #3b82f6; font-weight: bold;")
        elif index == 1:
            self.btn_data.setStyleSheet("border: none; color: #3b82f6; font-weight: bold;")
        else:
            self.btn_config.setStyleSheet("border: none; color: #3b82f6; font-weight: bold;")

    def open_details(self, card):
        """Open the detailed view dialog using the existing thread from the card."""
        title = card.lbl_title.text()
        existing_thread = card.thread
        video_source = card.source  # Get the video source path from the card

        # Pass self (MainWindow) so DetailsView can access all video cards
        dialog = VideoDetailDialog(self, video_title=title, video_source=video_source, thread=existing_thread, main_window=self)
        dialog.exec()

    def closeEvent(self, event):
        """Ensure all threads are stopped when window is closed."""
        for card in self.video_cards:
            card.stop_video()
        event.accept()
