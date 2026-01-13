"""
Database Data Viewer Page - Clean and Simple Design

显示数据库中的行人穿越记录，简洁易用的界面
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                               QTableWidget, QTableWidgetItem, QHeaderView, QComboBox,
                               QDateEdit, QDateTimeEdit, QLineEdit, QFrame, QMessageBox, QFileDialog,
                               QCheckBox, QGridLayout, QScrollArea, QCalendarWidget)
from PySide6.QtCore import Qt, QTimer, QDate, QDateTime, QThread, Signal
from PySide6.QtGui import QColor, QFont, QPixmap
from src.core.database import DatabaseManager
from src.core.config_manager import ConfigManager
from datetime import datetime, timedelta
import csv


class DataLoaderThread(QThread):
    """Background thread to load data from database"""
    data_loaded = Signal(list)

    def __init__(self, filters=None):
        super().__init__()
        self.db = DatabaseManager()
        self.filters = filters or {}

    def run(self):
        try:
            # Build query based on filters
            query = "SELECT * FROM crossing_events WHERE 1=1"
            params = []

            # DateTime range filter
            if self.filters.get('start_datetime'):
                query += " AND timestamp >= %s"
                params.append(self.filters['start_datetime'])
            if self.filters.get('end_datetime'):
                query += " AND timestamp <= %s"
                params.append(self.filters['end_datetime'])

            # Video ID filter (exact match from dropdown)
            if self.filters.get('video_id'):
                query += " AND video_id = %s"
                params.append(self.filters['video_id'])

            # Gender filter
            if self.filters.get('gender') and self.filters['gender'] != 'All':
                query += " AND gender = %s"
                params.append(self.filters['gender'])

            # Color filter
            if self.filters.get('color') and self.filters['color'] != 'All':
                query += " AND clothing_color = %s"
                params.append(self.filters['color'])

            # Mask filter
            if self.filters.get('mask') and self.filters['mask'] != 'All':
                query += " AND mask_status = %s"
                params.append(self.filters['mask'])

            # Handbag filter
            if self.filters.get('handbag') and self.filters['handbag'] != 'All':
                handbag_value = 1 if self.filters['handbag'] == 'With Handbag' else 0
                query += " AND handbag = %s"
                params.append(handbag_value)

            # Backpack filter
            if self.filters.get('backpack') and self.filters['backpack'] != 'All':
                backpack_value = 1 if self.filters['backpack'] == 'With Backpack' else 0
                query += " AND backpack = %s"
                params.append(backpack_value)

            # Sorting
            sort_column = self.filters.get('sort_column', 'timestamp')
            sort_order = self.filters.get('sort_order', 'DESC')
            query += f" ORDER BY {sort_column} {sort_order}"

            # Limit for performance
            limit = self.filters.get('limit', 1000)
            query += f" LIMIT {limit}"

            conn = self.db.connect()
            if conn:
                cursor = conn.cursor(dictionary=True)
                cursor.execute(query, params)
                results = cursor.fetchall()
                cursor.close()
                self.data_loaded.emit(results)
            else:
                self.data_loaded.emit([])
        except Exception as e:
            print(f"Data load error: {e}")
            self.data_loaded.emit([])


class StatCard(QFrame):
    """Refined minimalist statistics card with premium icon badges"""
    def __init__(self, title, value, icon, color):
        super().__init__()
        self.setFixedHeight(100)
        
        self.setStyleSheet(f"""
            StatCard {{
                background-color: white;
                border-radius: 12px;
                border: 1px solid #e2e8f0;
            }}
            StatCard:hover {{
                border-color: {color};
                background-color: #f8fafc;
            }}
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 15, 20, 15)
        layout.setSpacing(15)

        # Icon Badge (The colored pod)
        icon_container = QFrame()
        icon_container.setFixedSize(48, 48)
        # Use a very light/transparent version of the theme color
        icon_container.setStyleSheet(f"""
            QFrame {{
                background-color: {color}15; 
                border-radius: 14px;
                border: 2px solid {color}30;
            }}
        """)
        icon_layout = QVBoxLayout(icon_container)
        icon_layout.setContentsMargins(0, 0, 0, 0)
        icon_layout.setSpacing(0)
        
        icon_btn_lbl = QLabel(icon)
        icon_btn_lbl.setAlignment(Qt.AlignCenter)
        icon_btn_lbl.setStyleSheet(f"font-size: 22px; color: {color}; background: transparent; font-weight: bold;")
        icon_layout.addWidget(icon_btn_lbl)
        
        layout.addWidget(icon_container)

        # Content Layout
        content_layout = QVBoxLayout()
        content_layout.setSpacing(0)
        content_layout.setAlignment(Qt.AlignVCenter)
        
        # Title
        title_label = QLabel(title.upper())
        title_label.setStyleSheet(f"font-size: 10px; color: #64748b; font-weight: 800; letter-spacing: 1.2px;")
        content_layout.addWidget(title_label)

        # Value
        self.value_label = QLabel(value)
        self.value_label.setStyleSheet(f"font-size: 32px; color: #1e293b; font-weight: 900; letter-spacing: -0.5px;")
        self.value_label.setObjectName("value")
        content_layout.addWidget(self.value_label)
        
        layout.addLayout(content_layout)
        layout.addStretch()

    def update_value(self, value):
        self.value_label.setText(str(value))


class DataViewPage(QWidget):
    """数据库查看页面 - 简洁版"""

    def __init__(self):
        super().__init__()
        self.db = DatabaseManager()
        self.current_data = []
        self.loader_thread = None

        self.init_ui()

        # Auto-refresh every 60 seconds (Smart refresh)
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_current_view)
        self.refresh_timer.start(60000)

        # Initial load
        QTimer.singleShot(500, self.load_data)

    def refresh_page(self):
        """Public method to refresh the page state (e.g. video IDs)"""
        self.populate_video_ids()
        # Optionally reload data if needed, or wait for timer
        # self.load_data()

    def init_ui(self):
        """Initialize the UI"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(25, 25, 25, 25)
        main_layout.setSpacing(20)

        # === Header ===
        header_layout = QHBoxLayout()

        title = QLabel("Database Analytics")
        title.setStyleSheet("font-size: 28px; font-weight: 800; color: #1e293b; letter-spacing: -0.5px;")
        header_layout.addWidget(title)
        
        header_layout.addStretch()

        # Simple status indicator
        self.lbl_status = QLabel("Ready")
        self.lbl_status.setStyleSheet("color: #64748b; font-size: 11px; font-weight: 700; text-transform: uppercase;")
        header_layout.addWidget(self.lbl_status)

        # Export button
        btn_export = QPushButton("Export CSV")
        btn_export.setFixedHeight(34)
        btn_export.setCursor(Qt.PointingHandCursor)
        btn_export.setStyleSheet("""
            QPushButton {
                background-color: #3b82f6; color: white; border: none; border-radius: 6px;
                padding: 0px 15px; font-weight: 700; font-size: 12px;
            }
            QPushButton:hover { background-color: #2563eb; }
        """)
        btn_export.clicked.connect(self.generate_report)
        header_layout.addWidget(btn_export)

        main_layout.addLayout(header_layout)

        # === Statistics Cards ===
        stats_layout = QGridLayout()
        stats_layout.setSpacing(15)

        self.card_total = StatCard("Captured Logs", "0", "⚡", "#3b82f6") # Energy/Activity icon
        self.card_crossings = StatCard("Crossings", "0", "👣", "#10b981") # Walking footprints
        self.card_left = StatCard("Moving Left", "0", "←", "#f59e0b") # Standard Arrow
        self.card_right = StatCard("Moving Right", "0", "→", "#8b5cf6") # Standard Arrow

        stats_layout.addWidget(self.card_total, 0, 0)
        stats_layout.addWidget(self.card_crossings, 0, 1)
        stats_layout.addWidget(self.card_left, 0, 2)
        stats_layout.addWidget(self.card_right, 0, 3)

        main_layout.addLayout(stats_layout)

        # === Search Console (New Combined Card) ===
        self.search_card = QFrame()
        self.search_card.setStyleSheet("""
            QFrame#SearchCard {
                background: white;
                border: 1px solid #e2e8f0;
                border-radius: 12px;
            }
        """)
        self.search_card.setObjectName("SearchCard")
        
        # Main layout for the card
        search_card_layout = QVBoxLayout(self.search_card)
        search_card_layout.setContentsMargins(10, 10, 10, 10)
        search_card_layout.setSpacing(0)

        # 1. Top Bar (Basic Filters + Action Buttons)
        top_bar = QWidget()
        top_bar_layout = QHBoxLayout(top_bar)
        top_bar_layout.setContentsMargins(10, 5, 10, 5)
        top_bar_layout.setSpacing(20)

        # Source Selection
        source_group = QVBoxLayout()
        source_label = QLabel("CAMERA SOURCE")
        source_label.setStyleSheet("font-size: 10px; font-weight: 800; color: #94a3b8; letter-spacing: 0.5px;")
        self.combo_video_id = QComboBox()
        self.combo_video_id.setFixedWidth(220)
        source_group.addWidget(source_label)
        source_group.addWidget(self.combo_video_id)
        top_bar_layout.addLayout(source_group)

        # Time Period
        period_group = QVBoxLayout()
        period_label = QLabel("TIME WINDOW")
        period_label.setStyleSheet("font-size: 10px; font-weight: 800; color: #94a3b8; letter-spacing: 0.5px;")
        self.combo_period = QComboBox()
        self.combo_period.addItems(["Today", "Yesterday", "This Week", "This Month", "Last 7 Days", "Custom Range"])
        self.combo_period.setFixedWidth(140)
        period_group.addWidget(period_label)
        period_group.addWidget(self.combo_period)
        top_bar_layout.addLayout(period_group)

        # Custom Range (Hidden)
        self.custom_range_widget = QWidget()
        custom_range_layout = QHBoxLayout(self.custom_range_widget)
        custom_range_layout.setContentsMargins(0, 0, 0, 0)
        
        self.datetime_start = QDateTimeEdit(QDateTime.currentDateTime().addDays(-1))
        self.datetime_end = QDateTimeEdit(QDateTime.currentDateTime())
        for dt in [self.datetime_start, self.datetime_end]:
            dt.setCalendarPopup(True)
            dt.setDisplayFormat("MM-dd HH:mm")
            dt.setFixedWidth(110)
        
        range_group = QVBoxLayout()
        range_label = QLabel("CUSTOM RANGE")
        range_label.setStyleSheet("font-size: 10px; font-weight: 800; color: #94a3b8; letter-spacing: 0.5px;")
        range_inputs = QHBoxLayout()
        range_inputs.addWidget(self.datetime_start)
        range_inputs.addWidget(QLabel("-"))
        range_inputs.addWidget(self.datetime_end)
        range_group.addWidget(range_label)
        range_group.addLayout(range_inputs)
        custom_range_layout.addLayout(range_group)
        self.custom_range_widget.setVisible(False)
        top_bar_layout.addWidget(self.custom_range_widget)

        top_bar_layout.addStretch()

        # Action: Advanced Toggle
        self.btn_toggle_advanced = QPushButton("ADVANCED")
        self.btn_toggle_advanced.setFixedWidth(100)
        self.btn_toggle_advanced.setFixedHeight(38)
        self.btn_toggle_advanced.setCheckable(True)
        self.btn_toggle_advanced.setCursor(Qt.PointingHandCursor)
        self.btn_toggle_advanced.setStyleSheet("""
            QPushButton {
                background: #f8fafc; color: #64748b; border: 1px solid #e2e8f0; 
                border-radius: 6px; font-weight: 800; font-size: 10px;
            }
            QPushButton:hover { background: #f1f5f9; color: #1e293b; }
            QPushButton:checked { background: #eff6ff; color: #3b82f6; border-color: #3b82f6; }
        """)
        top_bar_layout.addWidget(self.btn_toggle_advanced)

        # Action: Refresh Button (Replaced RUN FILTER)
        self.btn_refresh = QPushButton("🔄 REFRESH DATA")
        self.btn_refresh.setFixedWidth(160)
        self.btn_refresh.setFixedHeight(38)
        self.btn_refresh.setCursor(Qt.PointingHandCursor)
        self.btn_refresh.setStyleSheet("""
            QPushButton {
                background: #3b82f6; color: white; border: none; border-radius: 6px;
                font-weight: 800; font-size: 11px; letter-spacing: 0.5px;
            }
            QPushButton:hover { background: #2563eb; }
            QPushButton:pressed { background: #1d4ed8; }
        """)
        top_bar_layout.addWidget(self.btn_refresh)

        search_card_layout.addWidget(top_bar)

        # 2. Advanced Panel (Collapsible Grid)
        self.advanced_panel = QFrame()
        self.advanced_panel.setVisible(False)
        self.advanced_panel.setStyleSheet("border-top: 1px solid #f1f5f9; background: #fafafa;")
        advanced_layout = QGridLayout(self.advanced_panel)
        advanced_layout.setContentsMargins(20, 20, 20, 20)
        advanced_layout.setSpacing(15)

        input_style = """
            QDateEdit, QDateTimeEdit, QLineEdit, QComboBox {
                padding: 8px 12px;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                background: #fdfdfd;
                font-size: 13px;
                color: #1e293b;
                min-height: 28px;
            }
            QDateEdit:focus, QDateTimeEdit:focus, QLineEdit:focus, QComboBox:focus {
                border-color: #3b82f6;
                background: white;
            }
            QComboBox::drop-down { border: none; width: 30px; }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #64748b;
                margin-right: 10px;
            }
        """

        label_style = "font-weight: 800; color: #475569; font-size: 10px; text-transform: uppercase; letter-spacing: 1px;"

        # Row 1 of advanced
        g_label = QLabel("GENDER")
        g_label.setStyleSheet(label_style)
        advanced_layout.addWidget(g_label, 0, 0)
        self.combo_gender = QComboBox()
        self.combo_gender.addItems(["All", "Male", "Female"])
        self.combo_gender.setStyleSheet(input_style)
        advanced_layout.addWidget(self.combo_gender, 0, 1)

        c_label = QLabel("COLOR")
        c_label.setStyleSheet(label_style)
        advanced_layout.addWidget(c_label, 0, 2)
        self.combo_color = QComboBox()
        self.combo_color.addItems(["All", "Black", "White", "Red", "Blue", "Green", "Yellow", "Orange", "Purple", "Pink", "Brown", "Gray"])
        self.combo_color.setStyleSheet(input_style)
        advanced_layout.addWidget(self.combo_color, 0, 3)

        m_label = QLabel("MASK")
        m_label.setStyleSheet(label_style)
        advanced_layout.addWidget(m_label, 0, 4)
        self.combo_mask = QComboBox()
        self.combo_mask.addItems(["All", "With Mask", "No Mask", "Mask Incorrect"])
        self.combo_mask.setStyleSheet(input_style)
        advanced_layout.addWidget(self.combo_mask, 0, 5)

        # Row 2 of advanced
        h_label = QLabel("HANDBAG")
        h_label.setStyleSheet(label_style)
        advanced_layout.addWidget(h_label, 1, 0)
        self.combo_handbag = QComboBox()
        self.combo_handbag.addItems(["All", "With Handbag", "No Handbag"])
        self.combo_handbag.setStyleSheet(input_style)
        advanced_layout.addWidget(self.combo_handbag, 1, 1)

        b_label = QLabel("BACKPACK")
        b_label.setStyleSheet(label_style)
        advanced_layout.addWidget(b_label, 1, 2)
        self.combo_backpack = QComboBox()
        self.combo_backpack.addItems(["All", "With Backpack", "No Backpack"])
        self.combo_backpack.setStyleSheet(input_style)
        advanced_layout.addWidget(self.combo_backpack, 1, 3)

        l_label = QLabel("MATCH LIMIT")
        l_label.setStyleSheet(label_style)
        advanced_layout.addWidget(l_label, 1, 4)
        self.combo_limit = QComboBox()
        self.combo_limit.addItems(["100", "500", "1000", "5000", "All"])
        self.combo_limit.setCurrentText("1000")
        self.combo_limit.setStyleSheet(input_style)
        advanced_layout.addWidget(self.combo_limit, 1, 5)

        search_card_layout.addWidget(self.advanced_panel)
        main_layout.addWidget(self.search_card)

        # --- SIGNALS ---
        self.btn_refresh.clicked.connect(self.refresh_current_view)
        self.btn_toggle_advanced.toggled.connect(self.advanced_panel.setVisible)
        self.combo_period.currentIndexChanged.connect(self.on_period_changed)
        
        # Also auto-update on choice if preferred, or just rely on the button
        # self.combo_period.currentIndexChanged.connect(self.load_data)
        # self.combo_video_id.currentIndexChanged.connect(self.load_data)

        # === Data Table ===
        self.table = QTableWidget()
        self.table.setColumnCount(13)
        self.table.setHorizontalHeaderLabels([
            "ID", "Time", "Location", "Line",
            "Left", "Right", "Color", "Gender", "Age", "Mask", "Handbag", "Backpack", "Video ID"
        ])

        # Table configuration
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)  # Location column stretches
        header.setDefaultAlignment(Qt.AlignLeft)

        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.ExtendedSelection)  # Enable multi-select
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSortingEnabled(True)
        self.table.setShowGrid(False)

        self.table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                background-color: white;
                gridline-color: #f1f5f9;
                font-size: 12px;
                color: #334155;
            }
            QHeaderView::section {
                background-color: #f8fafc;
                padding: 12px 10px;
                border: none;
                border-bottom: 2px solid #e2e8f0;
                font-weight: 800;
                color: #475569;
                font-size: 11px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            QTableWidget::item {
                padding: 12px 10px;
                border-bottom: 1px solid #f1f5f9;
            }
            QTableWidget::item:selected {
                background-color: #eff6ff;
                color: #2563eb;
                font-weight: 600;
            }
        """)

        self.table.setMinimumHeight(350)
        main_layout.addWidget(self.table)

        # Management Tools (Bottom Bar)
        manage_layout = QHBoxLayout()
        
        self.lbl_table_info = QLabel("Total logs matches current filters")
        self.lbl_table_info.setStyleSheet("color: #94a3b8; font-size: 11px;")
        manage_layout.addWidget(self.lbl_table_info)
        
        manage_layout.addStretch()

        from PySide6.QtWidgets import QMenu
        self.btn_manage = QPushButton("DATABASE CONTROLS")
        self.btn_manage.setFixedWidth(180)
        self.btn_manage.setFixedHeight(36)
        self.btn_manage.setCursor(Qt.PointingHandCursor)
        self.btn_manage.setStyleSheet("""
            QPushButton {
                background: #f8fafc; color: #475569; border: 1px solid #e2e8f0; 
                border-radius: 8px; font-weight: 800; font-size: 10px; letter-spacing: 0.5px;
            }
            QPushButton:hover { background: white; color: #1e293b; border-color: #cbd5e1; }
            QPushButton::menu-indicator { 
                image: none; 
            }
        """)
        
        manage_menu = QMenu(self)
        manage_menu.setStyleSheet("""
            QMenu {
                background-color: white;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 5px;
            }
            QMenu::item {
                padding: 10px 25px;
                border-radius: 5px;
                color: #475569;
                font-size: 12px;
                font-weight: 600;
            }
            QMenu::item:selected {
                background-color: #f1f5f9;
                color: #1e293b;
            }
            QMenu::separator {
                height: 1px;
                background: #f1f5f9;
                margin: 5px 10px;
            }
        """)
        
        act_clear = manage_menu.addAction("Clear Selection")
        act_clear.triggered.connect(self.table.clearSelection)
        
        manage_menu.addSeparator()
        
        act_del_sel = manage_menu.addAction("Delete Selected Rows")
        act_del_sel.triggered.connect(self.delete_selected_records)
        
        act_wipe = manage_menu.addAction("Wipe All Matching Data")
        act_wipe.triggered.connect(self.delete_filtered_records)
        
        self.btn_manage.setMenu(manage_menu)
        manage_layout.addWidget(self.btn_manage)

        main_layout.addLayout(manage_layout)
        
        # Build Video ID list
        self.populate_video_ids()

    def populate_video_ids(self):
        """Populate the video ID dropdown from both config and historical data in DB"""
        # CRITICAL: Always reload config to get latest sources added in other tabs
        cm = ConfigManager()
        cm.load_config() 
        
        self.combo_video_id.clear()
        self.combo_video_id.addItem("ALL SOURCES", "")
        
        # 1. Get current names from config for better display
        config_map = {}
        cm = ConfigManager()
        sources = cm.get("video_sources", [])
        for src in sources:
            vid = src.get("id")
            name = src.get("name", "Unknown Source")
            if vid:
                config_map[vid] = name
        
        # 2. Get all unique IDs that actually exist in the DB
        all_vids = self.db.get_unique_video_ids()
        
        # Merge - prioritize config order but include historical ones
        displayed_vids = set()
        
        # Add current ones from config
        for vid, name in config_map.items():
            self.combo_video_id.addItem(f"{vid} - {name}", vid)
            displayed_vids.add(vid)
            
        # Add historical ones not in config
        for vid in all_vids:
            if vid not in displayed_vids:
                self.combo_video_id.addItem(f"{vid} (Historical Data)", vid)
                displayed_vids.add(vid)

    def on_period_changed(self, index):
        """Handle period dropdown change"""
        period = self.combo_period.currentText()
        self.custom_range_widget.setVisible(period == "Custom Range")
        
        if period == "Today": self.set_today()
        elif period == "Yesterday": self.set_yesterday()
        elif period == "This Week": self.set_this_week()
        elif period == "This Month": self.set_this_month()
        elif period == "Last 7 Days": self.set_last_7days()
        # Custom Range does nothing, wait for user input

    def refresh_current_view(self):
        """Smart refresh: Updates 'now' for dynamic windows then loads."""
        period = self.combo_period.currentText()
        if period == "Today": self.set_today()
        elif period == "Yesterday": self.set_yesterday()
        elif period == "This Week": self.set_this_week()
        elif period == "This Month": self.set_this_month()
        elif period == "Last 7 Days": self.set_last_7days()
        else:
            # Custom Range - just load with existing values
            self.load_data()

    def set_today(self):
        """Set date range to today"""
        from PySide6.QtCore import QTime
        now = QDateTime.currentDateTime()
        today_start = QDateTime(now.date(), QTime(0, 0, 0))
        self.datetime_start.setDateTime(today_start)
        self.datetime_end.setDateTime(now)
        self.load_data()

    def set_yesterday(self):
        """Set date range to yesterday"""
        from PySide6.QtCore import QTime
        yesterday_date = QDateTime.currentDateTime().addDays(-1).date()
        yesterday_start = QDateTime(yesterday_date, QTime(0, 0, 0))
        yesterday_end = QDateTime(yesterday_date, QTime(23, 59, 59))
        self.datetime_start.setDateTime(yesterday_start)
        self.datetime_end.setDateTime(yesterday_end)
        self.load_data()

    def set_this_week(self):
        """Set date range to this week (Monday to now)"""
        from PySide6.QtCore import QTime
        now = QDateTime.currentDateTime()
        days_since_monday = now.date().dayOfWeek() - 1  # Monday = 1
        week_start_date = now.date().addDays(-days_since_monday)
        week_start = QDateTime(week_start_date, QTime(0, 0, 0))
        self.datetime_start.setDateTime(week_start)
        self.datetime_end.setDateTime(now)
        self.load_data()

    def set_last_7days(self):
        """Set date range to last 7 days"""
        from PySide6.QtCore import QTime
        now = QDateTime.currentDateTime()
        seven_days_ago_date = now.date().addDays(-7)
        seven_days_ago = QDateTime(seven_days_ago_date, QTime(0, 0, 0))
        self.datetime_start.setDateTime(seven_days_ago)
        self.datetime_end.setDateTime(now)
        self.load_data()

    def set_this_month(self):
        """Set date range to this month"""
        from PySide6.QtCore import QTime
        now = QDateTime.currentDateTime()
        month_start_date = now.date().addDays(-(now.date().day() - 1))
        month_start = QDateTime(month_start_date, QTime(0, 0, 0))
        self.datetime_start.setDateTime(month_start)
        self.datetime_end.setDateTime(now)
        self.load_data()

    def load_data(self):
        """Load data from database with filters"""
        if self.loader_thread and self.loader_thread.isRunning():
            return

        self.lbl_status.setText("⏳ Loading...")
        self.lbl_status.setStyleSheet("color: #3b82f6; font-size: 13px; font-weight: bold; padding: 5px;")

        gender_filter = None if self.combo_gender.currentText() == "All" else self.combo_gender.currentText()
        color_filter = None if self.combo_color.currentText() == "All" else self.combo_color.currentText()
        mask_filter = None if self.combo_mask.currentText() == "All" else self.combo_mask.currentText()
        handbag_filter = None if self.combo_handbag.currentText() == "All" else self.combo_handbag.currentText()
        backpack_filter = None if self.combo_backpack.currentText() == "All" else self.combo_backpack.currentText()
        limit = 999999 if self.combo_limit.currentText() == "All" else int(self.combo_limit.currentText())

        # Get selected video ID from combo box
        selected_video_id = self.combo_video_id.currentData()

        filters = {
            'start_datetime': self.datetime_start.dateTime().toString("yyyy-MM-dd HH:mm:ss"),
            'end_datetime': self.datetime_end.dateTime().toString("yyyy-MM-dd HH:mm:ss"),
            'video_id': selected_video_id if selected_video_id else "",
            'gender': gender_filter,
            'color': color_filter,
            'mask': mask_filter,
            'handbag': handbag_filter,
            'backpack': backpack_filter,
            'limit': limit,
            'sort_column': 'timestamp',
            'sort_order': 'DESC'
        }

        self.loader_thread = DataLoaderThread(filters)
        self.loader_thread.data_loaded.connect(self.on_data_loaded)
        self.loader_thread.start()

    def on_data_loaded(self, data):
        """Handle loaded data"""
        self.current_data = data

        # Update stats
        total_left = sum(d.get('count_left', 0) for d in data)
        total_right = sum(d.get('count_right', 0) for d in data)

        self.card_total.update_value(str(len(data)))
        self.card_crossings.update_value(str(total_left + total_right))
        self.card_left.update_value(str(total_left))
        self.card_right.update_value(str(total_right))

        # Clear and populate table
        self.table.setRowCount(0)
        self.table.setSortingEnabled(False)

        for row_data in data:
            row = self.table.rowCount()
            self.table.insertRow(row)

            # ID
            item_id = QTableWidgetItem(str(row_data.get('id', '')))
            item_id.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, 0, item_id)

            # Timestamp
            ts = row_data.get('timestamp', '')
            ts_str = ts.strftime('%m-%d %H:%M') if isinstance(ts, datetime) else str(ts)
            self.table.setItem(row, 1, QTableWidgetItem(ts_str))

            # Location
            self.table.setItem(row, 2, QTableWidgetItem(str(row_data.get('location', ''))))

            # Line
            self.table.setItem(row, 3, QTableWidgetItem(str(row_data.get('line_name', ''))))

            # Left count
            count_left = row_data.get('count_left', 0)
            item_left = QTableWidgetItem(str(count_left))
            item_left.setTextAlignment(Qt.AlignCenter)
            if count_left > 0:
                item_left.setForeground(QColor("#f59e0b"))
                item_left.setFont(QFont("Arial", 10, QFont.Bold))
            self.table.setItem(row, 4, item_left)

            # Right count
            count_right = row_data.get('count_right', 0)
            item_right = QTableWidgetItem(str(count_right))
            item_right.setTextAlignment(Qt.AlignCenter)
            if count_right > 0:
                item_right.setForeground(QColor("#8b5cf6"))
                item_right.setFont(QFont("Arial", 10, QFont.Bold))
            self.table.setItem(row, 5, item_right)

            # Color
            color = str(row_data.get('clothing_color', 'Unknown'))
            item_color = QTableWidgetItem(color)
            item_color.setTextAlignment(Qt.AlignCenter)

            color_map = {
                'Red': '#fecaca', 'Blue': '#bfdbfe', 'Green': '#bbf7d0',
                'Yellow': '#fef08a', 'Orange': '#fed7aa', 'Purple': '#e9d5ff',
                'Pink': '#fbcfe8', 'Black': '#e2e8f0', 'White': '#f8fafc',
                'Gray': '#d1d5db', 'Brown': '#d6bcad'
            }
            if color in color_map:
                item_color.setBackground(QColor(color_map[color]))
            self.table.setItem(row, 6, item_color)

            # Gender
            gender = str(row_data.get('gender', ''))
            if gender == 'Male':
                gender_text = '♂ M'
                gender_color = QColor("#2563eb")
            elif gender == 'Female':
                gender_text = '♀ F'
                gender_color = QColor("#db2777")
            else:
                gender_text = '—'
                gender_color = QColor("#94a3b8")

            item_gender = QTableWidgetItem(gender_text)
            item_gender.setForeground(gender_color)
            item_gender.setTextAlignment(Qt.AlignCenter)
            item_gender.setFont(QFont("Arial", 10, QFont.Bold))
            self.table.setItem(row, 7, item_gender)

            # Age
            age = row_data.get('age')
            age_text = str(age) if age else '—'
            item_age = QTableWidgetItem(age_text)
            item_age.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, 8, item_age)

            # Mask Status
            mask_status = row_data.get('mask_status', '')
            mask_text = mask_status if mask_status else '—'
            item_mask = QTableWidgetItem(mask_text)
            item_mask.setTextAlignment(Qt.AlignCenter)

            # Color code mask status
            if mask_status == 'With Mask':
                item_mask.setForeground(QColor("#10b981"))  # Green
                item_mask.setFont(QFont("Arial", 10, QFont.Bold))
            elif mask_status == 'No Mask':
                item_mask.setForeground(QColor("#ef4444"))  # Red
                item_mask.setFont(QFont("Arial", 10, QFont.Bold))
            elif mask_status == 'Mask Incorrect':
                item_mask.setForeground(QColor("#f59e0b"))  # Orange
                item_mask.setFont(QFont("Arial", 10, QFont.Bold))

            self.table.setItem(row, 9, item_mask)

            # Handbag
            handbag = row_data.get('handbag', 0)
            handbag_text = "👜 Yes" if handbag == 1 else "—"
            item_handbag = QTableWidgetItem(handbag_text)
            item_handbag.setTextAlignment(Qt.AlignCenter)
            if handbag == 1:
                item_handbag.setForeground(QColor("#8b5cf6"))  # Purple
                item_handbag.setFont(QFont("Arial", 10, QFont.Bold))
            self.table.setItem(row, 10, item_handbag)

            # Backpack
            backpack = row_data.get('backpack', 0)
            backpack_text = "🎒 Yes" if backpack == 1 else "—"
            item_backpack = QTableWidgetItem(backpack_text)
            item_backpack.setTextAlignment(Qt.AlignCenter)
            if backpack == 1:
                item_backpack.setForeground(QColor("#10b981"))  # Green
                item_backpack.setFont(QFont("Arial", 10, QFont.Bold))
            self.table.setItem(row, 11, item_backpack)

            # Video ID
            video_id = str(row_data.get('video_id', ''))
            item_video = QTableWidgetItem(video_id)
            item_video.setFont(QFont("Courier", 9))
            item_video.setForeground(QColor("#64748b"))
            self.table.setItem(row, 12, item_video)

        self.table.setSortingEnabled(True)

        # Update status
        if len(data) > 0:
            self.lbl_status.setText(f"✅ Loaded {len(data)} records | {datetime.now().strftime('%H:%M:%S')}")
            self.lbl_status.setStyleSheet("color: #10b981; font-size: 13px; font-weight: bold; padding: 5px;")
        else:
            self.lbl_status.setText("⚠ No records found")
            self.lbl_status.setStyleSheet("color: #f59e0b; font-size: 13px; font-weight: bold; padding: 5px;")

    def generate_report(self):
        """Generate HTML report with statistics and detailed data"""
        if not self.current_data:
            QMessageBox.warning(self, "No Data", "Please load data first.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Generate Report",
            f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            "HTML Files (*.html)"
        )

        if not file_path:
            return

        try:
            # Calculate statistics
            total_records = len(self.current_data)
            total_crossings = sum(row.get('count_left', 0) + row.get('count_right', 0) for row in self.current_data)
            total_left = sum(row.get('count_left', 0) for row in self.current_data)
            total_right = sum(row.get('count_right', 0) for row in self.current_data)

            # Gender breakdown
            gender_counts = {}
            for row in self.current_data:
                gender = row.get('gender', 'Unknown')
                gender_counts[gender] = gender_counts.get(gender, 0) + 1

            # Color breakdown
            color_counts = {}
            for row in self.current_data:
                color = row.get('clothing_color', 'Unknown')
                color_counts[color] = color_counts.get(color, 0) + 1

            # Mask breakdown
            mask_counts = {}
            for row in self.current_data:
                mask = row.get('mask_status', 'Unknown')
                if not mask:
                    mask = 'Unknown'
                mask_counts[mask] = mask_counts.get(mask, 0) + 1

            # Video ID breakdown
            video_counts = {}
            for row in self.current_data:
                vid = row.get('video_id', 'Unknown')
                video_counts[vid] = video_counts.get(vid, 0) + 1

            # Get date range
            start_date = self.datetime_start.dateTime().toString("yyyy-MM-dd HH:mm")
            end_date = self.datetime_end.dateTime().toString("yyyy-MM-dd HH:mm")

            # Generate HTML report
            html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Crowd Detection Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        .header h1 {{
            font-size: 32px;
            margin-bottom: 10px;
        }}
        .header p {{
            opacity: 0.9;
            font-size: 16px;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 30px;
            background: #f8fafc;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .stat-card h3 {{
            color: #64748b;
            font-size: 14px;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .stat-card .value {{
            font-size: 36px;
            font-weight: bold;
            color: #1e293b;
        }}
        .section {{
            padding: 30px;
        }}
        .section h2 {{
            color: #1e293b;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }}
        .breakdown {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .breakdown-card {{
            background: #f8fafc;
            padding: 20px;
            border-radius: 12px;
        }}
        .breakdown-card h3 {{
            color: #475569;
            margin-bottom: 15px;
            font-size: 16px;
        }}
        .breakdown-item {{
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #e2e8f0;
        }}
        .breakdown-item:last-child {{
            border-bottom: none;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
        }}
        th {{
            background: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        td {{
            padding: 12px;
            border-bottom: 1px solid #e2e8f0;
        }}
        tr:hover {{
            background: #f8fafc;
        }}
        .footer {{
            background: #1e293b;
            color: white;
            padding: 20px;
            text-align: center;
        }}
        .filter-info {{
            background: #eff6ff;
            border-left: 4px solid #3b82f6;
            padding: 15px 20px;
            margin-bottom: 20px;
            border-radius: 4px;
        }}
        .filter-info strong {{
            color: #1e40af;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Crowd Detection Report</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>

        <div class="filter-info" style="margin: 20px 30px;">
            <strong>Report Period:</strong> {start_date} to {end_date}
        </div>

        <div class="summary">
            <div class="stat-card">
                <h3>Total Records</h3>
                <div class="value">{total_records}</div>
            </div>
            <div class="stat-card">
                <h3>Total Crossings</h3>
                <div class="value">{total_crossings}</div>
            </div>
            <div class="stat-card">
                <h3>Left Direction</h3>
                <div class="value" style="color: #f59e0b;">{total_left}</div>
            </div>
            <div class="stat-card">
                <h3>Right Direction</h3>
                <div class="value" style="color: #8b5cf6;">{total_right}</div>
            </div>
        </div>

        <div class="section">
            <h2>📈 Statistics Breakdown</h2>
            <div class="breakdown">
                <div class="breakdown-card">
                    <h3>👥 Gender Distribution</h3>
                    {''.join(f'<div class="breakdown-item"><span>{gender}</span><span><strong>{count}</strong></span></div>' for gender, count in sorted(gender_counts.items(), key=lambda x: x[1], reverse=True))}
                </div>
                <div class="breakdown-card">
                    <h3>🎨 Clothing Colors</h3>
                    {''.join(f'<div class="breakdown-item"><span>{color}</span><span><strong>{count}</strong></span></div>' for color, count in sorted(color_counts.items(), key=lambda x: x[1], reverse=True)[:10])}
                </div>
                <div class="breakdown-card">
                    <h3>😷 Mask Status</h3>
                    {''.join(f'<div class="breakdown-item"><span>{mask}</span><span><strong>{count}</strong></span></div>' for mask, count in sorted(mask_counts.items(), key=lambda x: x[1], reverse=True))}
                </div>
                <div class="breakdown-card">
                    <h3>📹 Video Sources</h3>
                    {''.join(f'<div class="breakdown-item"><span>{vid}</span><span><strong>{count}</strong></span></div>' for vid, count in sorted(video_counts.items(), key=lambda x: x[1], reverse=True))}
                </div>
            </div>
        </div>

        <div class="section">
            <h2>📋 Detailed Records</h2>
            <table>
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Video ID</th>
                        <th>Location</th>
                        <th>Line</th>
                        <th>Left</th>
                        <th>Right</th>
                        <th>Gender</th>
                        <th>Age</th>
                        <th>Color</th>
                        <th>Mask</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(f'''
                    <tr>
                        <td>{row.get("timestamp", "").strftime("%Y-%m-%d %H:%M:%S") if isinstance(row.get("timestamp"), datetime) else row.get("timestamp", "")}</td>
                        <td>{row.get("video_id", "")}</td>
                        <td>{row.get("location", "")}</td>
                        <td>{row.get("line_name", "")}</td>
                        <td>{row.get("count_left", 0)}</td>
                        <td>{row.get("count_right", 0)}</td>
                        <td>{row.get("gender", "")}</td>
                        <td>{row.get("age", "")}</td>
                        <td>{row.get("clothing_color", "")}</td>
                        <td>{row.get("mask_status", "")}</td>
                    </tr>
                    ''' for row in self.current_data[:500])}
                </tbody>
            </table>
            {f'<p style="margin-top: 15px; color: #64748b; font-style: italic;">Showing first 500 of {total_records} records</p>' if total_records > 500 else ''}
        </div>

        <div class="footer">
            <p>🤖 Generated by Crowd Detection System | © 2025</p>
        </div>
    </div>
</body>
</html>
"""

            # Write HTML file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            # Show success message and offer to open
            reply = QMessageBox.question(
                self,
                "Report Generated",
                f"Report generated successfully with {total_records} records!\n\nDo you want to open it now?",
                QMessageBox.Yes | QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                import webbrowser
                import os
                webbrowser.open('file://' + os.path.abspath(file_path))

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Report generation failed:\n{str(e)}")

    def delete_selected_records(self):
        """Delete selected rows from the table"""
        selected_rows = self.table.selectedItems()
        if not selected_rows:
            QMessageBox.warning(self, "No Selection", "Please select records to delete.")
            return

        # Get unique row indices
        selected_row_indices = set()
        for item in selected_rows:
            selected_row_indices.add(item.row())

        # Get IDs from selected rows
        ids_to_delete = []
        for row_idx in selected_row_indices:
            id_item = self.table.item(row_idx, 0)  # ID is in column 0
            if id_item:
                ids_to_delete.append(int(id_item.text()))

        # Confirm deletion
        reply = QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to delete {len(ids_to_delete)} selected record(s)?\n\nThis action cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        # Delete from database
        db = DatabaseManager()
        success_count = 0
        for event_id in ids_to_delete:
            if db.delete_event(event_id):
                success_count += 1

        # Show result
        if success_count > 0:
            QMessageBox.information(
                self,
                "Deletion Complete",
                f"Successfully deleted {success_count} record(s)."
            )
            # Reload data
            self.load_data()
        else:
            QMessageBox.warning(
                self,
                "Deletion Failed",
                "Failed to delete records. Please try again."
            )

    def delete_filtered_records(self):
        """Delete all records matching current filters"""
        if not self.current_data:
            QMessageBox.warning(self, "No Data", "No records to delete. Please apply filters first.")
            return

        # Confirm deletion
        reply = QMessageBox.question(
            self,
            "Confirm Bulk Deletion",
            f"Are you sure you want to delete ALL {len(self.current_data)} filtered record(s)?\n\n"
            f"This will delete all records matching your current filters.\n"
            f"This action cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        # Build filters from current UI state
        gender_filter = None if self.combo_gender.currentText() == "All" else self.combo_gender.currentText()
        color_filter = None if self.combo_color.currentText() == "All" else self.combo_color.currentText()
        mask_filter = None if self.combo_mask.currentText() == "All" else self.combo_mask.currentText()
        handbag_filter = None if self.combo_handbag.currentText() == "All" else self.combo_handbag.currentText()
        backpack_filter = None if self.combo_backpack.currentText() == "All" else self.combo_backpack.currentText()
        selected_video_id = self.combo_video_id.currentData()

        filters = {
            'start_datetime': self.datetime_start.dateTime().toString("yyyy-MM-dd HH:mm:ss"),
            'end_datetime': self.datetime_end.dateTime().toString("yyyy-MM-dd HH:mm:ss"),
            'video_id': selected_video_id if selected_video_id else "",
            'gender': gender_filter,
            'color': color_filter,
            'mask': mask_filter,
            'handbag': handbag_filter,
            'backpack': backpack_filter,
        }

        # Delete from database
        db = DatabaseManager()
        deleted_count = db.delete_events_by_filter(filters)

        # Show result
        if deleted_count:
            QMessageBox.information(
                self,
                "Deletion Complete",
                f"Successfully deleted {deleted_count} record(s)."
            )
            # Reload data
            self.load_data()
        else:
            QMessageBox.warning(
                self,
                "Deletion Failed",
                "Failed to delete records. Please try again."
            )
