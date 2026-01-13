import os
import datetime
import time
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                               QScrollArea, QFrame, QDateEdit, QGridLayout, QMessageBox,
                               QSizePolicy, QComboBox, QCheckBox)
from PySide6.QtCore import Qt, QDateTime, QDate, QUrl, QSize, QTime
from PySide6.QtGui import QDesktopServices, QFont, QIcon, QColor, QPixmap

class RecordingCard(QFrame):
    """Modern card representation of a video recording with thumbnail"""
    def __init__(self, file_info, play_callback, delete_callback, select_callback):
        super().__init__()
        self.file_info = file_info
        self.play_idx = play_callback
        self.delete_idx = delete_callback
        self.select_callback = select_callback
        self.isSelected = False
        self.init_ui()

    def init_ui(self):
        self.setFixedSize(280, 260)
        self.setCursor(Qt.PointingHandCursor)
        self.setObjectName("RecordingCard")
        
        # Main Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Thumbnail Area (Top)
        self.thumb_container = QLabel()
        self.thumb_container.setFixedSize(280, 150)
        self.thumb_container.setAlignment(Qt.AlignCenter)
        self.thumb_container.setStyleSheet("background-color: #0f172a; border-top-left-radius: 12px; border-top-right-radius: 12px;")
        
        thumb_path = self.file_info.get('thumb_path')
        if thumb_path and os.path.exists(thumb_path):
            pix = QPixmap(thumb_path)
            self.thumb_container.setPixmap(pix.scaled(280, 150, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation))
        else:
            self.thumb_container.setText("PREVIEW UNAVAILABLE")
            self.thumb_container.setStyleSheet("color: #475569; background: #1e293b; font-size: 10px; font-weight: 800; border-top-left-radius: 12px; border-top-right-radius: 12px;")
        
        layout.addWidget(self.thumb_container)
        
        # Content Info
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(15, 12, 15, 15)
        content_layout.setSpacing(6)

        # Source + Size
        top_row = QHBoxLayout()
        source_name = self.file_info['source'].upper()
        source_lbl = QLabel(source_name)
        source_lbl.setStyleSheet("font-weight: 800; font-size: 10px; color: #64748b; letter-spacing: 1px;")
        top_row.addWidget(source_lbl)
        top_row.addStretch()
        
        size_mb = self.file_info['size'] / (1024 * 1024)
        size_lbl = QLabel(f"{size_mb:.1f} MB")
        size_lbl.setStyleSheet("color: #94a3b8; font-size: 10px; font-weight: 600;")
        top_row.addWidget(size_lbl)
        content_layout.addLayout(top_row)

        # Smart Time Range (Main Title)
        time_display = self.file_info['time'].strftime("%Y-%m-%d  %H:%M:%S")
        fname = self.file_info['name']
        if "__TO__" in fname:
            try:
                parts = fname.replace(".mp4", "").split("_")
                start_hms = parts[1]
                end_hms = parts[5]
                time_display = f"{start_hms[:2]}:{start_hms[2:4]} ➔ {end_hms[:2]}:{end_hms[2:4]} ({parts[0][4:6]}/{parts[0][6:8]})"
            except: pass

        time_lbl = QLabel(time_display)
        time_lbl.setStyleSheet("color: #1e293b; font-size: 13px; font-weight: 700;")
        content_layout.addWidget(time_lbl)

        # Action Buttons
        actions = QHBoxLayout()
        actions.setSpacing(8)
        
        btn_play = QPushButton("VIEW FOOTAGE")
        btn_play.setFixedHeight(32)
        btn_play.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        btn_play.setCursor(Qt.PointingHandCursor)
        btn_play.clicked.connect(lambda: self.play_idx(self.file_info['path']))
        btn_play.setStyleSheet("""
            QPushButton {
                background-color: #3b82f6; color: white; border-radius: 6px; 
                font-weight: 800; font-size: 10px; border: none;
            }
            QPushButton:hover { background-color: #2563eb; }
        """)
        
        btn_del = QPushButton("DELETE")
        btn_del.setFixedSize(70, 32)
        btn_del.setCursor(Qt.PointingHandCursor)
        btn_del.clicked.connect(lambda: self.delete_idx(self.file_info, self.file_info['path']))
        btn_del.setStyleSheet("""
            QPushButton {
                background-color: #f1f5f9; color: #ef4444; border-radius: 6px; 
                font-weight: 800; font-size: 10px; border: none;
            }
            QPushButton:hover { background-color: #fee2e2; }
        """)
        
        actions.addWidget(btn_play)
        actions.addWidget(btn_del)
        content_layout.addLayout(actions)
        layout.addWidget(content)

        # Selection Overlay Checkbox (custom styling)
        self.check_select = QCheckBox(self)
        self.check_select.setFixedSize(24, 24)
        self.check_select.move(10, 10)
        self.check_select.setCursor(Qt.PointingHandCursor)
        self.check_select.stateChanged.connect(self.on_check_changed)
        self.check_select.setStyleSheet("""
            QCheckBox::indicator {
                width: 18px; height: 18px;
                border: 2px solid rgba(255,255,255,0.8); border-radius: 5px;
                background-color: rgba(0,0,0,0.4);
            }
            QCheckBox::indicator:checked {
                background-color: #3b82f6; border: 2px solid #3b82f6;
                image: url(none); /* Optional: add a checkmark icon if available */
            }
        """)

        self.setStyleSheet("""
            #RecordingCard {
                background-color: white;
                border: 1px solid #e2e8f0;
                border-radius: 12px;
            }
        """)

    def on_check_changed(self, state):
        self.isSelected = (state == Qt.Checked.value)
        if self.isSelected:
            self.setStyleSheet("""
                #RecordingCard {
                    background-color: #eff6ff;
                    border: 2px solid #3b82f6;
                    border-radius: 12px;
                }
            """)
        else:
            self.setStyleSheet("""
                #RecordingCard {
                    background-color: white;
                    border: 1px solid #e2e8f0;
                    border-radius: 12px;
                }
            """)
        self.select_callback()

    def set_selected(self, selected):
        self.check_select.setChecked(selected)

class RecordingsPage(QWidget):
    def __init__(self):
        super().__init__()
        self.recordings_dir = os.path.abspath("recordings")
        if not os.path.exists(self.recordings_dir):
            os.makedirs(self.recordings_dir)
            
        self.cards = [] # Store list of card widgets
        self.init_ui()

        self.refresh_recordings()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(25)

        # === Header Section ===
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 10)

        title_container = QVBoxLayout()
        title = QLabel("Evidence Gallery")
        title.setStyleSheet("font-size: 28px; font-weight: 800; color: #1e293b; letter-spacing: -0.5px;")
        subtitle = QLabel("Review and manage AI detection recordings")
        subtitle.setStyleSheet("font-size: 13px; color: #64748b; font-weight: 500;")
        title_container.addWidget(title)
        title_container.addWidget(subtitle)
        header_layout.addLayout(title_container)

        header_layout.addStretch()

        # Quick Actions
        self.btn_refresh = QPushButton("REFRESH GALLERY")
        self.btn_refresh.setFixedSize(160, 36)
        self.btn_refresh.setCursor(Qt.PointingHandCursor)
        self.btn_refresh.clicked.connect(self.refresh_recordings)
        self.btn_refresh.setStyleSheet("""
            QPushButton {
                background: #f8fafc; color: #475569; border: 1px solid #e2e8f0; 
                border-radius: 6px; font-weight: 800; font-size: 10px; letter-spacing: 0.5px;
            }
            QPushButton:hover { background: white; color: #1e293b; border-color: #cbd5e1; }
        """)
        header_layout.addWidget(self.btn_refresh)
        main_layout.addLayout(header_layout)

        # === Search Console (Unified Style) ===
        self.search_card = QFrame()
        self.search_card.setStyleSheet("background: white; border: 1px solid #e2e8f0; border-radius: 12px;")
        search_layout = QHBoxLayout(self.search_card)
        search_layout.setContentsMargins(15, 10, 15, 10)
        search_layout.setSpacing(25)

        # Date Range Group
        date_group = QVBoxLayout()
        date_lbl = QLabel("DATE RANGE")
        date_lbl.setStyleSheet("font-size: 10px; font-weight: 800; color: #94a3b8; letter-spacing: 1px;")
        date_inputs = QHBoxLayout()
        self.date_start = QDateEdit(QDate.currentDate().addDays(-7))
        self.date_end = QDateEdit(QDate.currentDate())
        for d in [self.date_start, self.date_end]:
            d.setCalendarPopup(True)
            d.setFixedWidth(120)
            d.setFixedHeight(32)
            d.setStyleSheet("""
                QDateEdit { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; font-weight: 600; font-size: 12px; }
                QDateEdit:focus { border-color: #3b82f6; background: white; }
            """)
        date_inputs.addWidget(self.date_start)
        date_inputs.addWidget(QLabel("➔"))
        date_inputs.addWidget(self.date_end)
        date_group.addWidget(date_lbl)
        date_group.addLayout(date_inputs)
        search_layout.addLayout(date_group)

        # Source Group
        source_group = QVBoxLayout()
        source_lbl = QLabel("CAMERA SOURCE")
        source_lbl.setStyleSheet("font-size: 10px; font-weight: 800; color: #94a3b8; letter-spacing: 1px;")
        self.combo_source = QComboBox()
        self.combo_source.setFixedWidth(180)
        self.combo_source.setFixedHeight(32)
        self.combo_source.setStyleSheet("""
            QComboBox { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; font-weight: 600; padding-left: 10px; }
            QComboBox:focus { border-color: #3b82f6; background: white; }
        """)
        source_group.addWidget(source_lbl)
        source_group.addWidget(self.combo_source)
        search_layout.addLayout(source_group)

        search_layout.addStretch()

        # Bulk Actions (Hidden by default, shown when items selected)
        self.bulk_container = QWidget()
        self.bulk_container.setVisible(False)
        bulk_layout = QHBoxLayout(self.bulk_container)
        bulk_layout.setContentsMargins(0, 0, 0, 0)
        
        self.check_all = QCheckBox("SELECT ALL")
        self.check_all.setStyleSheet("font-weight: 800; font-size: 10px; color: #475569; letter-spacing: 1px;")
        bulk_layout.addWidget(self.check_all)
        
        self.btn_bulk_delete = QPushButton("DELETE SELECTED")
        self.btn_bulk_delete.setFixedHeight(38)
        self.btn_bulk_delete.setFixedWidth(150)
        self.btn_bulk_delete.setStyleSheet("""
            QPushButton { background: #ef4444; color: white; border-radius: 6px; font-weight: 800; font-size: 11px; }
            QPushButton:hover { background: #dc2626; }
        """)
        bulk_layout.addWidget(self.btn_bulk_delete)
        search_layout.addWidget(self.bulk_container)

        # Main RUN Button
        self.btn_apply = QPushButton("RUN FILTER")
        self.btn_apply.setFixedSize(140, 38)
        self.btn_apply.setCursor(Qt.PointingHandCursor)
        self.btn_apply.clicked.connect(self.refresh_recordings)
        self.btn_apply.setStyleSheet("""
            QPushButton { background: #0f172a; color: white; border-radius: 6px; font-weight: 800; font-size: 11px; letter-spacing: 0.5px; }
            QPushButton:hover { background: #1e293b; }
        """)
        search_layout.addWidget(self.btn_apply)

        main_layout.addWidget(self.search_card)

        # Connect signals
        self.check_all.stateChanged.connect(self.toggle_select_all)
        self.btn_bulk_delete.clicked.connect(self.bulk_delete)

        # === Grid Area ===
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        
        self.grid_container = QWidget()
        self.grid_container.setStyleSheet("background: transparent;")
        self.grid_layout = QGridLayout(self.grid_container)
        self.grid_layout.setSpacing(25)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.grid_layout.setAlignment(Qt.AlignLeft | Qt.AlignTop) # Pushes items to the left
        self.scroll.setWidget(self.grid_container)
        main_layout.addWidget(self.scroll)

    def resizeEvent(self, event):
        """Handle resize to rearrange grid"""
        super().resizeEvent(event)
        
        # Grid logic
        self.rearrange_grid()

    def rearrange_grid(self):
        """Calculate and apply the best grid layout for recordings based on current width"""
        if not hasattr(self, 'cards') or not self.cards:
            return
            
        # Total available width
        available_width = self.width() - 80 
        card_width = 280 + 25 
        
        cols = max(1, available_width // card_width)
        
        # We need to clear and re-add to maintain left-alignment properly in QGridLayout
        for i, card in enumerate(self.cards):
            self.grid_layout.addWidget(card, i // cols, i % cols)
            
        # Add a stretchable column and row to keep everything tight
        self.grid_layout.setColumnStretch(cols, 1)
        self.grid_layout.setRowStretch(len(self.cards) // cols + 1, 1)

    def refresh_recordings(self):
        # Clear existing grid
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        self.cards = []
        self.bulk_container.setVisible(False)
        self.check_all.blockSignals(True)
        self.check_all.setChecked(False)
        self.check_all.blockSignals(False)

        if not os.path.exists(self.recordings_dir):
            return

        selected_source = self.combo_source.currentText()
        
        # Calculate full day range
        start_ts = QDateTime(self.date_start.date(), QTime(0, 0, 0)).toSecsSinceEpoch()
        end_ts = QDateTime(self.date_end.date(), QTime(23, 59, 59)).toSecsSinceEpoch()

        files = []
        available_sources = set(["All Sources"])
        
        for root, dirs, filenames in os.walk(self.recordings_dir):
            for f in filenames:
                if f.endswith(".mp4"):
                    path = os.path.join(root, f)
                    stat = os.stat(path)
                    mtime = stat.st_mtime
                    
                    if start_ts <= mtime <= end_ts:
                        rel_path = os.path.relpath(path, self.recordings_dir)
                        path_parts = rel_path.split(os.sep)
                        source_name = "General"
                        
                        if len(path_parts) > 1:
                            source_name = path_parts[0].replace("_", " ")
                        elif "__" in f:
                            source_name = f.split("__")[0].replace("_", " ")
                        
                        available_sources.add(source_name)

                        # Filter by Source
                        if selected_source != "All Sources" and source_name != selected_source:
                            continue
                        
                        # Check for thumbnail
                        thumb_path = path.replace(".mp4", "_thumb.jpg")
                        if not os.path.exists(thumb_path):
                            thumb_path = None
                        
                        files.append({
                            "source": source_name,
                            "name": f,
                            "path": os.path.abspath(path),
                            "time": datetime.datetime.fromtimestamp(mtime),
                            "size": stat.st_size,
                            "thumb_path": thumb_path
                        })
        
        # Sort by time descending
        files.sort(key=lambda x: x["time"], reverse=True)

        # Update ComboBox items if list changed
        current_selection = self.combo_source.currentText()
        new_source_list = sorted(list(available_sources))
        
        # Only update if the list is actually different (to prevent infinite loops if we were using signals)
        if [self.combo_source.itemText(i) for i in range(self.combo_source.count())] != new_source_list:
            self.combo_source.blockSignals(True)
            self.combo_source.clear()
            self.combo_source.addItems(new_source_list)
            idx = self.combo_source.findText(current_selection)
            if idx >= 0:
                self.combo_source.setCurrentIndex(idx)
            else:
                self.combo_source.setCurrentIndex(0)
            self.combo_source.blockSignals(False)

        if not files:
            empty_lbl = QLabel("No recordings found matching the filters.")
            empty_lbl.setStyleSheet("color: #94a3b8; font-size: 16px; margin: 50px;")
            self.grid_layout.addWidget(empty_lbl, 0, 0)
            return

        # Populating grid (using fluid rearrangement)
        for i, file_info in enumerate(files):
            card = RecordingCard(file_info, self.play_video, self.delete_video, self.update_bulk_ui)
            self.cards.append(card)
        
        self.rearrange_grid()

    def update_bulk_ui(self):
        """Show/Hide bulk delete button based on selection"""
        selected_count = sum(1 for card in self.cards if card.isSelected)
        self.bulk_container.setVisible(selected_count > 0)
        self.btn_bulk_delete.setText(f"DELETE SELECTED ({selected_count})")

    def toggle_select_all(self, state):
        # Convert state to bool (Qt.Checked is a CheckState enum member)
        selected = (state == Qt.Checked or state == 2)
        for card in self.cards:
            card.set_selected(selected)

    def bulk_delete(self):
        selected_cards = [c for c in self.cards if c.isSelected]
        if not selected_cards: return
        
        reply = QMessageBox.question(self, "Bulk Delete", 
                                   f"Are you sure you want to permanently delete {len(selected_cards)} items?",
                                   QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            for card in selected_cards:
                path = card.file_info['path']
                try:
                    if os.path.exists(path):
                        os.remove(path)
                    tp = card.file_info.get("thumb_path")
                    if tp and os.path.exists(tp):
                        os.remove(tp)
                    
                    # Clean up empty source folder
                    parent_dir = os.path.dirname(path)
                    if os.path.exists(parent_dir) and parent_dir != self.recordings_dir:
                        if not os.listdir(parent_dir):
                            os.rmdir(parent_dir)
                except: pass
            
            self.refresh_recordings()

    def play_video(self, path):
        if os.path.exists(path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def delete_video(self, file_info, path):
        reply = QMessageBox.question(self, "Delete Recording", 
                                   "Are you sure you want to permanently delete this evidence?",
                                   QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            try:
                # Delete video
                if os.path.exists(path):
                    os.remove(path)
                
                # Delete thumbnail if exists
                tp = file_info.get("thumb_path")
                if tp and os.path.exists(tp):
                    os.remove(tp)
                
                # Clean up empty source folder
                parent_dir = os.path.dirname(path)
                if os.path.exists(parent_dir) and parent_dir != self.recordings_dir:
                    # Check if directory is empty (no files or subfolders)
                    if not os.listdir(parent_dir):
                        try:
                            os.rmdir(parent_dir)
                            print(f"[RECORDINGS] 🗑️ Cleaned up empty folder: {os.path.basename(parent_dir)}")
                        except: pass
                    
                self.refresh_recordings()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not delete: {e}")
