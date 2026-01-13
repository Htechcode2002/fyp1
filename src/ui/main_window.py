from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
                               QPushButton, QFrame, QScrollArea, QStackedWidget, QSpacerItem, QSizePolicy)
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QPixmap
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

        self.setWindowTitle("AI VISION CCTV SYSTEM")
        self.setWindowIcon(QIcon("assets/logo.png"))
        self.resize(1280, 850)
        # Clean Professional Light Theme
        self.setStyleSheet("background-color: #ffffff;") 

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
        header.setFixedHeight(80)
        header.setStyleSheet("background-color: white; border-bottom: 2px solid #f1f5f9;")
        header_layout = QHBoxLayout(header)
        
        self.lbl_logo_img = QLabel()
        pix = QPixmap("assets/logo.png")
        # Scaled to 64x64
        self.lbl_logo_img.setPixmap(pix.scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.lbl_logo_img.setStyleSheet("border: none; margin-left: 15px;")

        lbl_app_name = QLabel("AI VISION")
        lbl_app_name.setStyleSheet("font-weight: 800; font-size: 22px; color: #1e293b; letter-spacing: 1.5px;")
        
        self.btn_dashboard = QPushButton("DASHBOARD")
        self.btn_data = QPushButton("DATABASE")
        self.btn_recordings = QPushButton("RECORDINGS")
        self.btn_config = QPushButton("CONFIG")
        
        self.btn_dashboard.setCursor(Qt.PointingHandCursor)
        self.btn_data.setCursor(Qt.PointingHandCursor)
        self.btn_recordings.setCursor(Qt.PointingHandCursor)
        self.btn_config.setCursor(Qt.PointingHandCursor)
        
        self.btn_dashboard.clicked.connect(lambda: self.switch_page(0))
        self.btn_data.clicked.connect(lambda: self.switch_page(1))
        self.btn_recordings.clicked.connect(lambda: self.switch_page(3))
        self.btn_config.clicked.connect(lambda: self.switch_page(2))

        # Modern minimalist button style
        button_style = """
            QPushButton {
                border: none; 
                color: #64748b; 
                font-weight: 800; 
                font-size: 13px;
                padding: 10px 18px;
                background-color: transparent;
                border-radius: 6px;
                margin: 5px 0px;
            }
            QPushButton:hover {
                color: #1e293b;
                background-color: #f1f5f9;
            }
            QPushButton[active="true"] {
                color: #ffffff;
                background-color: #3b82f6;
            }
        """
        for btn in [self.btn_dashboard, self.btn_data, self.btn_recordings, self.btn_config]:
            btn.setStyleSheet(button_style)

        header_layout.addWidget(self.lbl_logo_img)
        header_layout.addWidget(lbl_app_name)
        header_layout.addStretch()
        header_layout.addWidget(self.btn_dashboard)
        header_layout.addSpacing(10)
        header_layout.addWidget(self.btn_data)
        header_layout.addSpacing(10)
        header_layout.addWidget(self.btn_recordings)
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

        # Page 3: Recordings
        from src.ui.recordings_page import RecordingsPage
        recordings_page = RecordingsPage()
        self.stack.addWidget(recordings_page)

        main_layout.addWidget(self.stack)

        # Initial Refresh
        self.switch_page(0)
        self.refresh_dashboard()

    def create_dashboard_page(self):
        """Creates the dashboard container with a static watermark and a scrollable grid."""
        container = QWidget()
        container.setStyleSheet("background-color: #ffffff;") # Reverted to white
        container_layout = QGridLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)

        # --- Watermark (Subtle Background, Static) ---
        self.lbl_watermark = QLabel(container)
        watermark_pix = QPixmap("assets/system_watermark.png")
        if not watermark_pix.isNull():
            self.lbl_watermark.setPixmap(watermark_pix.scaled(600, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.lbl_watermark.setAlignment(Qt.AlignCenter)
        self.lbl_watermark.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.lbl_watermark.setStyleSheet("background: transparent;")
        
        # Dashboard Scroll Area (on top)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setStyleSheet("background: transparent;") # Crucial to see watermark below
        
        content_widget = QWidget()
        content_widget.setObjectName("DashboardPage")
        content_widget.setStyleSheet("background-color: transparent;") # Transparent to show watermark

        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(50, 40, 50, 40)
        content_layout.setSpacing(15)

        # Dashboard Header
        title_container = QWidget()
        title_container.setStyleSheet("background: transparent;")
        title_layout = QHBoxLayout(title_container)
        title_layout.setContentsMargins(0, 0, 0, 0)
        
        lbl_dash_title = QLabel("System Dashboard")
        lbl_dash_title.setStyleSheet("font-size: 28px; font-weight: 800; color: #1e293b; letter-spacing: -0.5px;")
        
        title_layout.addWidget(lbl_dash_title)
        title_layout.addStretch()
        content_layout.addWidget(title_container)
        
        self.lbl_subtitle = QLabel("Evaluating sources...")
        self.lbl_subtitle.setStyleSheet("color: #64748b; font-weight: 600;")
        content_layout.addWidget(self.lbl_subtitle)

        # Container for Video Cards
        self.cards_container = QWidget()
        self.cards_layout = QGridLayout(self.cards_container)
        self.cards_layout.setContentsMargins(0, 0, 0, 0)
        self.cards_layout.setSpacing(25)
        self.cards_layout.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        
        content_layout.addWidget(self.cards_container)
        content_layout.addStretch()

        scroll_area.setWidget(content_widget)
        
        # Add both to the same grid cell of the container
        container_layout.addWidget(self.lbl_watermark, 0, 0)
        container_layout.addWidget(scroll_area, 0, 0)
        
        return container

    def resizeEvent(self, event):
        """Handle resize to center watermark and rearrange grid"""
        super().resizeEvent(event)
        
        # Center watermark in its container
        if hasattr(self, 'lbl_watermark'):
            w, h = 600, 600 # We scaled it to 600x600
            cw, ch = self.width(), self.height() - 80 # Minus header height
            # Note: geometry is relative to 'container', which fills self.stack
            self.lbl_watermark.setGeometry((self.stack.width()-w)//2, (self.stack.height()-h)//2, w, h)

        if not hasattr(self, 'stack'):
            return
            
        current_idx = self.stack.currentIndex()
        if current_idx == 0:
            self.rearrange_grid()
        elif current_idx == 3:
            recordings_page = self.stack.widget(3)
            if hasattr(recordings_page, 'rearrange_grid'):
                recordings_page.rearrange_grid()

    def rearrange_grid(self):
        """Calculate and apply the best grid layout based on current width"""
        if not hasattr(self, 'video_cards') or not self.video_cards:
            return
            
        # Total available width for cards (with some margin breathing room)
        available_width = self.width() - 100 # Adjust based on sidebar/margins
        card_width = 420 + 25 # Card width + spacing
        
        # Calculate how many columns can fit (minimum 1)
        cols = max(1, available_width // card_width)
        
        # Remove and re-add widgets to the grid
        for i, card in enumerate(self.video_cards):
            self.cards_layout.addWidget(card, i // cols, i % cols)

    def refresh_dashboard(self):
        """Smart reload video sources from config. Only restarts changed sources."""
        cm = ConfigManager()
        sources = cm.load_config().get("video_sources", []) # Reload from disk
        
        # 1. Identify which IDs are in the new config
        new_ids = [src.get("id") for src in sources]
        
        # 2. Remove cards that are no longer in the config
        cards_to_keep = []
        for card in self.video_cards:
            if card.video_id in new_ids:
                cards_to_keep.append(card)
            else:
                card.stop_video()
                card.setParent(None)
                card.deleteLater()
        
        # Clear the layout before re-adding
        for i in reversed(range(self.cards_layout.count())): 
            widget_to_remove = self.cards_layout.itemAt(i).widget()
            if widget_to_remove:
                self.cards_layout.removeWidget(widget_to_remove)
                # widget_to_remove.setParent(None) # Not strictly necessary if deleteLater is called or if it's just removed from layout

        self.video_cards = cards_to_keep
        
        # 3. Add or update cards from the new config
        existing_ids = [card.video_id for card in self.video_cards]
        
        cols = 3 # Wrap every 3 cards
        
        for i, src in enumerate(sources):
            video_id = src.get("id")
            path = src.get("path", "")
            location = src.get("location", "")
            name = src.get("name", "Unknown Source")
            display_title = location if location else name
            
            danger_threshold = src.get("danger_threshold", 100)
            loitering_threshold = src.get("loitering_threshold", 5.0)
            fall_threshold = src.get("fall_threshold", 2.0)

            if video_id in existing_ids:
                # Update existing card
                card = next(c for c in self.video_cards if c.video_id == video_id)
                self.cards_layout.addWidget(card, i // cols, i % cols) # Re-add to correct grid position
                
                card.lbl_title.setText(display_title)
                
                # If path changed, we MUST restart
                if card.source != path:
                    print(f"[DASHBOARD] 🔄 Path changed for {video_id}, restarting...")
                    if path.isdigit():
                        card.start_video(int(path), location=location, video_id=video_id, danger_threshold=danger_threshold, loitering_threshold=loitering_threshold, fall_threshold=fall_threshold)
                    else:
                        card.start_video(path, location=location, video_id=video_id, danger_threshold=danger_threshold, loitering_threshold=loitering_threshold, fall_threshold=fall_threshold)
                else:
                    # Just update thresholds on the existing detector
                    if card.thread and card.thread.detector:
                        card.thread.detector.danger_threshold = danger_threshold
                        card.thread.detector.loitering_threshold = loitering_threshold
                        card.thread.detector.fall_threshold = fall_threshold
            else:
                # Create NEW card
                card = VideoCard(title=display_title, video_id=video_id)
                card.setFixedWidth(420)
                card.btn_details.clicked.connect(lambda _, c=card: self.open_details(c))
                
                # Add to video_cards list
                self.video_cards.append(card)
                
                # Grid placement
                self.cards_layout.addWidget(card, i // cols, i % cols)
                if path:
                    if path.isdigit():
                        card.start_video(int(path), location=location, video_id=video_id, danger_threshold=danger_threshold, loitering_threshold=loitering_threshold, fall_threshold=fall_threshold)
                    else:
                        card.start_video(path, location=location, video_id=video_id, danger_threshold=danger_threshold, loitering_threshold=loitering_threshold, fall_threshold=fall_threshold)

        # 4. Final step: Arrange everything in a fluid grid
        self.rearrange_grid()

        # Update subtitle
        count = len(self.video_cards)
        self.lbl_subtitle.setText(f"{count} video source{'s' if count != 1 else ''} available")
            
    def switch_page(self, index):
        """Switch stacked pages and update navigation UI"""
        self.stack.setCurrentIndex(index)

        # Refresh screens if necessary
        if index == 0:
            self.refresh_dashboard()
        elif index == 1:
            data_page = self.stack.widget(1)
            if hasattr(data_page, 'refresh_page'):
                data_page.refresh_page()
        elif index == 3:
            recordings_page = self.stack.widget(3)
            if hasattr(recordings_page, 'refresh_recordings'):
                recordings_page.refresh_recordings()

        # Premium Light Navigation Styling
        common_style = """
            QPushButton {
                border: none; 
                color: #64748b; 
                font-weight: 800; 
                font-size: 13px;
                padding: 10px 18px;
                background-color: transparent;
                border-radius: 6px;
                margin: 5px 0px;
                letter-spacing: 0.5px;
            }
            QPushButton:hover {
                color: #1e293b;
                background-color: #f1f5f9;
            }
        """
        active_style = """
            QPushButton {
                border: none; 
                color: #ffffff; 
                font-weight: 800; 
                font-size: 13px;
                padding: 10px 18px;
                background-color: #3b82f6;
                border-radius: 6px;
                margin: 5px 0px;
                letter-spacing: 0.5px;
            }
        """
        
        # Reset all buttons
        self.btn_dashboard.setStyleSheet(common_style)
        self.btn_data.setStyleSheet(common_style)
        self.btn_recordings.setStyleSheet(common_style)
        self.btn_config.setStyleSheet(common_style)

        # Apply active style based on index
        if index == 0: self.btn_dashboard.setStyleSheet(active_style)
        elif index == 1: self.btn_data.setStyleSheet(active_style)
        elif index == 3: self.btn_recordings.setStyleSheet(active_style)
        elif index == 2: self.btn_config.setStyleSheet(active_style)

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
