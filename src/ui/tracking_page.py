"""
Cross-Camera Person Tracking Page

允许用户：
1. 加载多个视频
2. 选择要追踪的目标人物
3. 在所有视频中搜索该人物
4. 显示搜索结果
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                               QListWidget, QListWidgetItem, QFileDialog, QFrame,
                               QScrollArea, QGridLayout, QMessageBox, QProgressBar)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QImage, QPixmap
import cv2
import numpy as np
from src.core.detection import VideoDetector
from src.core.multi_camera_tracker import MultiCameraTracker
from src.core.reid_extractor import ReIDFeatureExtractor
import time


class VideoSearchThread(QThread):
    """后台线程用于搜索视频中的目标人物"""
    progress = Signal(int, int)  # (current_video, total_videos)
    found = Signal(int, float, np.ndarray, dict)  # (video_idx, timestamp, frame, detection)
    finished = Signal()

    def __init__(self, video_paths, target_feature, reid_extractor, similarity_threshold=0.6):
        super().__init__()
        self.video_paths = video_paths
        self.target_feature = target_feature
        self.reid_extractor = reid_extractor
        self.similarity_threshold = similarity_threshold
        self.running = True

    def run(self):
        """搜索所有视频"""
        for video_idx, video_path in enumerate(self.video_paths):
            if not self.running:
                break

            self.progress.emit(video_idx + 1, len(self.video_paths))

            # 打开视频
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25

            # 创建检测器（不使用跨镜头追踪）
            detector = VideoDetector(camera_id=f"search_{video_idx}")

            frame_count = 0
            while self.running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 每30帧检测一次（加速搜索）
                if frame_count % 30 != 0:
                    frame_count += 1
                    continue

                # 检测人物
                detections, _ = detector.detect(frame, tracking_enabled=False)

                # 对每个检测到的人提取特征并比对
                for det in detections:
                    if det['cls_id'] != 0:  # 只检测人
                        continue

                    # 提取ReID特征
                    box = det['box']
                    feature = self.reid_extractor.extract_features(frame, box)

                    # 计算相似度
                    similarity = self.reid_extractor.compute_similarity(
                        self.target_feature, feature
                    )

                    # 如果相似度超过阈值，发送结果
                    if similarity >= self.similarity_threshold:
                        timestamp = frame_count / fps
                        self.found.emit(video_idx, timestamp, frame.copy(), det)

                frame_count += 1

            cap.release()

        self.finished.emit()

    def stop(self):
        """停止搜索"""
        self.running = False


class CrossCameraTrackingPage(QWidget):
    """跨摄像头追踪页面"""

    def __init__(self):
        super().__init__()

        self.video_paths = []  # 已加载的视频路径
        self.target_feature = None  # 目标人物的ReID特征
        self.reid_extractor = ReIDFeatureExtractor()
        self.search_results = []  # 搜索结果

        self.init_ui()

    def init_ui(self):
        """初始化UI"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # 标题
        title = QLabel("跨视频人物追踪")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #1e293b;")
        main_layout.addWidget(title)

        # 说明
        desc = QLabel("1. 加载多个视频  2. 点击选择目标人物  3. 开始搜索")
        desc.setStyleSheet("color: #64748b; font-size: 14px;")
        main_layout.addWidget(desc)

        # 内容区域
        content_layout = QHBoxLayout()

        # === 左侧：视频列表 ===
        left_panel = QFrame()
        left_panel.setStyleSheet("background-color: white; border-radius: 8px;")
        left_panel.setMaximumWidth(350)
        left_layout = QVBoxLayout(left_panel)

        # 视频列表标题
        video_title = QLabel("已加载视频")
        video_title.setStyleSheet("font-weight: bold; font-size: 16px;")
        left_layout.addWidget(video_title)

        # 添加视频按钮
        btn_add_video = QPushButton("+ 添加视频")
        btn_add_video.setStyleSheet("""
            QPushButton {
                background-color: #3b82f6;
                color: white;
                border-radius: 6px;
                padding: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2563eb;
            }
        """)
        btn_add_video.clicked.connect(self.add_videos)
        left_layout.addWidget(btn_add_video)

        # 视频列表
        self.video_list = QListWidget()
        self.video_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                padding: 5px;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #f0f0f0;
            }
            QListWidget::item:selected {
                background-color: #eff6ff;
                color: #1e293b;
            }
        """)
        left_layout.addWidget(self.video_list)

        # 清空按钮
        btn_clear = QPushButton("清空列表")
        btn_clear.setStyleSheet("""
            QPushButton {
                background-color: #ef4444;
                color: white;
                border-radius: 6px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #dc2626;
            }
        """)
        btn_clear.clicked.connect(self.clear_videos)
        left_layout.addWidget(btn_clear)

        content_layout.addWidget(left_panel)

        # === 右侧：操作和结果 ===
        right_panel = QFrame()
        right_panel.setStyleSheet("background-color: white; border-radius: 8px;")
        right_layout = QVBoxLayout(right_panel)

        # 选择目标区域
        target_section = QLabel("选择追踪目标")
        target_section.setStyleSheet("font-weight: bold; font-size: 16px;")
        right_layout.addWidget(target_section)

        target_info = QLabel("从任意视频中点击选择要追踪的人物")
        target_info.setStyleSheet("color: #64748b; font-size: 13px;")
        right_layout.addWidget(target_info)

        # 选择目标按钮
        btn_select_target = QPushButton("从视频中选择目标")
        btn_select_target.setStyleSheet("""
            QPushButton {
                background-color: #10b981;
                color: white;
                border-radius: 6px;
                padding: 12px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #059669;
            }
        """)
        btn_select_target.clicked.connect(self.select_target_from_video)
        right_layout.addWidget(btn_select_target)

        # 目标预览
        self.target_preview = QLabel("未选择目标")
        self.target_preview.setAlignment(Qt.AlignCenter)
        self.target_preview.setStyleSheet("""
            border: 2px dashed #cbd5e1;
            border-radius: 8px;
            padding: 20px;
            background-color: #f8fafc;
            color: #94a3b8;
            min-height: 150px;
        """)
        right_layout.addWidget(self.target_preview)

        # 搜索按钮
        self.btn_search = QPushButton("开始搜索")
        self.btn_search.setEnabled(False)
        self.btn_search.setStyleSheet("""
            QPushButton {
                background-color: #f59e0b;
                color: white;
                border-radius: 6px;
                padding: 15px;
                font-weight: bold;
                font-size: 16px;
            }
            QPushButton:hover:enabled {
                background-color: #d97706;
            }
            QPushButton:disabled {
                background-color: #d1d5db;
            }
        """)
        self.btn_search.clicked.connect(self.start_search)
        right_layout.addWidget(self.btn_search)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #cbd5e1;
                border-radius: 4px;
                text-align: center;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #3b82f6;
            }
        """)
        right_layout.addWidget(self.progress_bar)

        # 搜索结果
        result_title = QLabel("搜索结果")
        result_title.setStyleSheet("font-weight: bold; font-size: 16px; margin-top: 20px;")
        right_layout.addWidget(result_title)

        # 结果滚动区域
        result_scroll = QScrollArea()
        result_scroll.setWidgetResizable(True)
        result_scroll.setStyleSheet("border: 1px solid #e0e0e0; border-radius: 4px;")

        self.result_widget = QWidget()
        self.result_layout = QVBoxLayout(self.result_widget)
        self.result_layout.setAlignment(Qt.AlignTop)
        result_scroll.setWidget(self.result_widget)

        right_layout.addWidget(result_scroll)

        content_layout.addWidget(right_panel)

        main_layout.addLayout(content_layout)

    def add_videos(self):
        """添加视频文件"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "选择视频文件",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )

        if file_paths:
            for path in file_paths:
                if path not in self.video_paths:
                    self.video_paths.append(path)
                    item = QListWidgetItem(f"📹 {path.split('/')[-1]}")
                    item.setToolTip(path)
                    self.video_list.addItem(item)

    def clear_videos(self):
        """清空视频列表"""
        self.video_paths.clear()
        self.video_list.clear()
        self.target_feature = None
        self.target_preview.setText("未选择目标")
        self.target_preview.setPixmap(QPixmap())
        self.btn_search.setEnabled(False)

    def select_target_from_video(self):
        """从视频中选择目标人物"""
        if not self.video_paths:
            QMessageBox.warning(self, "警告", "请先添加视频！")
            return

        # 打开目标选择对话框
        from src.ui.target_selector import TargetSelectorDialog
        dialog = TargetSelectorDialog(self.video_paths[0], self)

        if dialog.exec():
            # 获取选中的人物特征
            self.target_feature = dialog.get_target_feature()

            if self.target_feature is not None:
                # 显示目标预览
                target_img = dialog.get_target_image()
                if target_img is not None:
                    # 转换为QPixmap
                    height, width, channel = target_img.shape
                    bytes_per_line = 3 * width
                    q_img = QImage(target_img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped().copy()
                    pixmap = QPixmap.fromImage(q_img)

                    # 缩放并显示
                    scaled_pixmap = pixmap.scaled(200, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.target_preview.setPixmap(scaled_pixmap)
                    self.target_preview.setText("")

                # 启用搜索按钮
                self.btn_search.setEnabled(True)

    def start_search(self):
        """开始搜索"""
        if not self.video_paths:
            QMessageBox.warning(self, "警告", "请先添加视频！")
            return

        if self.target_feature is None:
            QMessageBox.warning(self, "警告", "请先选择目标人物！")
            return

        # 清空之前的结果
        self.clear_results()

        # 显示进度条
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(len(self.video_paths))

        # 禁用搜索按钮
        self.btn_search.setEnabled(False)
        self.btn_search.setText("搜索中...")

        # 创建搜索线程
        self.search_thread = VideoSearchThread(
            self.video_paths,
            self.target_feature,
            self.reid_extractor,
            similarity_threshold=0.5  # 可调整
        )

        self.search_thread.progress.connect(self.on_search_progress)
        self.search_thread.found.connect(self.on_target_found)
        self.search_thread.finished.connect(self.on_search_finished)

        self.search_thread.start()

    def on_search_progress(self, current, total):
        """更新搜索进度"""
        self.progress_bar.setValue(current)

    def on_target_found(self, video_idx, timestamp, frame, detection):
        """找到目标"""
        # 添加到结果列表
        self.search_results.append({
            'video_idx': video_idx,
            'video_path': self.video_paths[video_idx],
            'timestamp': timestamp,
            'frame': frame,
            'detection': detection
        })

        # 显示结果
        self.add_result_item(video_idx, timestamp, frame, detection)

    def on_search_finished(self):
        """搜索完成"""
        self.progress_bar.setVisible(False)
        self.btn_search.setEnabled(True)
        self.btn_search.setText("开始搜索")

        # 显示结果统计
        QMessageBox.information(
            self,
            "搜索完成",
            f"搜索完成！在 {len(self.search_results)} 个位置找到目标人物。"
        )

    def add_result_item(self, video_idx, timestamp, frame, detection):
        """添加一个搜索结果项"""
        result_frame = QFrame()
        result_frame.setStyleSheet("""
            QFrame {
                background-color: #f8fafc;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                padding: 10px;
                margin: 5px;
            }
        """)
        result_layout = QHBoxLayout(result_frame)

        # 缩略图
        box = detection['box']
        x1, y1, x2, y2 = box
        crop = frame[y1:y2, x1:x2]

        if crop.size > 0:
            height, width, channel = crop.shape
            bytes_per_line = 3 * width
            q_img = QImage(crop.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped().copy()
            pixmap = QPixmap.fromImage(q_img)
            scaled = pixmap.scaled(80, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            thumb = QLabel()
            thumb.setPixmap(scaled)
            result_layout.addWidget(thumb)

        # 信息
        info_layout = QVBoxLayout()
        video_name = self.video_paths[video_idx].split('/')[-1]
        info_layout.addWidget(QLabel(f"<b>视频:</b> {video_name}"))
        info_layout.addWidget(QLabel(f"<b>时间:</b> {int(timestamp//60)}:{int(timestamp%60):02d}"))

        color = detection.get('shirt_color', 'Unknown')
        info_layout.addWidget(QLabel(f"<b>颜色:</b> {color}"))

        result_layout.addLayout(info_layout)
        result_layout.addStretch()

        self.result_layout.addWidget(result_frame)

    def clear_results(self):
        """清空搜索结果"""
        while self.result_layout.count():
            item = self.result_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.search_results.clear()
