"""
实时跨镜头人物追踪页面

功能：
1. 加载多个视频源（实时或录像）
2. 从任意视频中选择目标人物
3. 实时监控所有视频，发现目标时告警
"""

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                               QFrame, QGridLayout, QScrollArea, QMessageBox, QListWidget,
                               QListWidgetItem)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QImage, QPixmap
import cv2
import numpy as np
from src.core.detection import VideoDetector
from src.core.reid_extractor import ReIDFeatureExtractor
import time


class VideoMonitorThread(QThread):
    """视频监控线程"""
    frame_ready = Signal(int, np.ndarray, list)  # (camera_idx, frame, detections)
    target_found = Signal(int, str, np.ndarray, dict)  # (camera_idx, camera_name, frame, detection)

    def __init__(self, camera_idx, video_source, camera_name):
        super().__init__()
        self.camera_idx = camera_idx
        self.video_source = video_source
        self.camera_name = camera_name
        self.running = False
        self.target_feature = None
        self.reid_extractor = ReIDFeatureExtractor()
        self.detector = VideoDetector(camera_id=f"live_{camera_idx}")
        self.similarity_threshold = 0.5

    def set_target(self, feature, threshold=0.5):
        """设置追踪目标"""
        self.target_feature = feature
        self.similarity_threshold = threshold

    def clear_target(self):
        """清除追踪目标"""
        self.target_feature = None

    def run(self):
        """运行监控"""
        cap = cv2.VideoCapture(self.video_source)

        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                # 视频结束，重新开始
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # 检测人物
            detections, _ = self.detector.detect(frame, tracking_enabled=True)

            # 如果有目标，检查是否找到
            if self.target_feature is not None:
                for det in detections:
                    if det['cls_id'] != 0:  # 只检测人
                        continue

                    # 提取特征
                    box = det['box']
                    feature = self.reid_extractor.extract_features(frame, box)

                    # 计算相似度
                    similarity = self.reid_extractor.compute_similarity(
                        self.target_feature, feature
                    )

                    # 如果相似度超过阈值，发送告警
                    if similarity >= self.similarity_threshold:
                        det['similarity'] = similarity
                        self.target_found.emit(
                            self.camera_idx,
                            self.camera_name,
                            frame.copy(),
                            det
                        )

            # 发送帧用于显示
            self.frame_ready.emit(self.camera_idx, frame, detections)

            # 控制帧率
            time.sleep(0.03)  # ~30 FPS

        cap.release()

    def stop(self):
        """停止监控"""
        self.running = False


class CameraWidget(QFrame):
    """单个摄像头显示组件"""

    def __init__(self, camera_idx, camera_name):
        super().__init__()
        self.camera_idx = camera_idx
        self.camera_name = camera_name

        self.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
            }
        """)

        layout = QVBoxLayout(self)

        # 摄像头名称
        name_label = QLabel(f"📹 {camera_name}")
        name_label.setStyleSheet("font-weight: bold; font-size: 14px; padding: 5px;")
        layout.addWidget(name_label)

        # 视频显示区域
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("""
            background-color: #1e293b;
            border-radius: 4px;
            min-height: 200px;
        """)
        self.video_label.setText("加载中...")
        layout.addWidget(self.video_label)

        # 状态标签
        self.status_label = QLabel("正常监控")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #64748b; padding: 5px;")
        layout.addWidget(self.status_label)

    def update_frame(self, frame):
        """更新显示帧"""
        # 转换为QPixmap
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_img)

        # 缩放
        scaled = pixmap.scaled(400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled)

    def show_alert(self):
        """显示告警"""
        self.setStyleSheet("""
            QFrame {
                background-color: #fef2f2;
                border: 3px solid #ef4444;
                border-radius: 8px;
            }
        """)
        self.status_label.setText("⚠️ 发现目标！")
        self.status_label.setStyleSheet("color: #ef4444; font-weight: bold; padding: 5px;")

    def clear_alert(self):
        """清除告警"""
        self.setStyleSheet("""
            QFrame {
                background-color: white;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
            }
        """)
        self.status_label.setText("正常监控")
        self.status_label.setStyleSheet("color: #64748b; padding: 5px;")


class LiveTrackingPage(QWidget):
    """实时跨镜头追踪页面"""

    def __init__(self):
        super().__init__()

        self.video_sources = []  # [(source_path, name), ...]
        self.monitor_threads = []
        self.camera_widgets = {}
        self.target_feature = None
        self.reid_extractor = ReIDFeatureExtractor()

        # 告警记录
        self.alerts = []

        self.init_ui()

    def init_ui(self):
        """初始化UI"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # === 顶部控制栏 ===
        control_panel = QFrame()
        control_panel.setStyleSheet("background-color: white; border-radius: 8px; padding: 15px;")
        control_layout = QHBoxLayout(control_panel)

        # 标题
        title = QLabel("实时跨镜头追踪")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        control_layout.addWidget(title)

        control_layout.addStretch()

        # 选择目标按钮
        self.btn_select_target = QPushButton("选择追踪目标")
        self.btn_select_target.setStyleSheet("""
            QPushButton {
                background-color: #10b981;
                color: white;
                border-radius: 6px;
                padding: 10px 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #059669;
            }
        """)
        self.btn_select_target.clicked.connect(self.select_target)
        control_layout.addWidget(self.btn_select_target)

        # 开始监控按钮
        self.btn_start = QPushButton("开始监控")
        self.btn_start.setEnabled(False)
        self.btn_start.setStyleSheet("""
            QPushButton {
                background-color: #3b82f6;
                color: white;
                border-radius: 6px;
                padding: 10px 20px;
                font-weight: bold;
            }
            QPushButton:hover:enabled {
                background-color: #2563eb;
            }
            QPushButton:disabled {
                background-color: #d1d5db;
            }
        """)
        self.btn_start.clicked.connect(self.start_monitoring)
        control_layout.addWidget(self.btn_start)

        # 停止监控按钮
        self.btn_stop = QPushButton("停止监控")
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet("""
            QPushButton {
                background-color: #ef4444;
                color: white;
                border-radius: 6px;
                padding: 10px 20px;
                font-weight: bold;
            }
            QPushButton:hover:enabled {
                background-color: #dc2626;
            }
            QPushButton:disabled {
                background-color: #d1d5db;
            }
        """)
        self.btn_stop.clicked.connect(self.stop_monitoring)
        control_layout.addWidget(self.btn_stop)

        main_layout.addWidget(control_panel)

        # === 目标预览 ===
        target_frame = QFrame()
        target_frame.setStyleSheet("background-color: white; border-radius: 8px; padding: 10px;")
        target_layout = QHBoxLayout(target_frame)

        target_layout.addWidget(QLabel("<b>追踪目标:</b>"))

        self.target_preview = QLabel("未选择")
        self.target_preview.setAlignment(Qt.AlignCenter)
        self.target_preview.setStyleSheet("""
            border: 2px dashed #cbd5e1;
            border-radius: 4px;
            padding: 10px;
            background-color: #f8fafc;
            min-width: 80px;
            max-width: 80px;
            min-height: 100px;
        """)
        target_layout.addWidget(self.target_preview)

        target_layout.addStretch()

        # 告警列表
        target_layout.addWidget(QLabel("<b>告警记录:</b>"))

        self.alert_list = QListWidget()
        self.alert_list.setMaximumHeight(120)
        self.alert_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #e0e0e0;
                border-radius: 4px;
            }
            QListWidget::item {
                padding: 5px;
            }
        """)
        target_layout.addWidget(self.alert_list)

        main_layout.addWidget(target_frame)

        # === 摄像头网格 ===
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: none;")

        camera_container = QWidget()
        self.camera_grid = QGridLayout(camera_container)
        self.camera_grid.setSpacing(15)

        scroll.setWidget(camera_container)
        main_layout.addWidget(scroll)

        # 加载Dashboard中的视频源
        self.load_video_sources()

    def load_video_sources(self):
        """加载视频源 - 从本地配置或让用户手动添加"""
        # 尝试从配置文件加载预设视频
        from src.core.config_manager import ConfigManager
        import os

        config = ConfigManager()

        # 预设的视频源（如果存在）
        preset_videos = config.get("preset_videos", [])

        # 如果没有预设，显示提示
        if not preset_videos:
            # 扫描常见视频目录
            video_dirs = ["videos", "samples", "."]
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv']

            for video_dir in video_dirs:
                if os.path.exists(video_dir):
                    for file in os.listdir(video_dir):
                        if any(file.lower().endswith(ext) for ext in video_extensions):
                            video_path = os.path.join(video_dir, file)
                            name = os.path.splitext(file)[0]
                            preset_videos.append({
                                'path': video_path,
                                'name': name
                            })

        # 加载视频源
        for idx, source in enumerate(preset_videos[:6]):  # 最多6个
            if isinstance(source, dict):
                source_path = source.get('path', '')
                location = source.get('name', f"Camera {idx + 1}")
            else:
                source_path = source
                location = f"Camera {idx + 1}"

            if os.path.exists(source_path):
                self.video_sources.append((source_path, location))

                # 创建摄像头组件
                widget = CameraWidget(idx, location)
                self.camera_widgets[idx] = widget

                # 添加到网格（2列）
                row = idx // 2
                col = idx % 2
                self.camera_grid.addWidget(widget, row, col)

        # 如果没有找到任何视频，显示提示
        if not self.video_sources:
            info_label = QLabel("📹 没有找到视频源\n\n请在当前目录或videos文件夹中放置视频文件")
            info_label.setAlignment(Qt.AlignCenter)
            info_label.setStyleSheet("""
                background-color: #fef2f2;
                border: 2px dashed #ef4444;
                border-radius: 8px;
                padding: 40px;
                color: #991b1b;
                font-size: 14px;
            """)
            self.camera_grid.addWidget(info_label, 0, 0, 1, 2)

    def select_target(self):
        """选择追踪目标"""
        if not self.video_sources:
            QMessageBox.warning(self, "警告", "没有可用的视频源！请先在Dashboard中添加视频。")
            return

        # 打开目标选择对话框
        from src.ui.target_selector import TargetSelectorDialog

        # 使用第一个视频源
        dialog = TargetSelectorDialog(self.video_sources[0][0], self)

        if dialog.exec():
            self.target_feature = dialog.get_target_feature()

            if self.target_feature is not None:
                # 显示目标预览
                target_img = dialog.get_target_image()
                if target_img is not None:
                    height, width, channel = target_img.shape
                    bytes_per_line = 3 * width
                    q_img = QImage(target_img.data, width, height, bytes_per_line,
                                 QImage.Format_RGB888).rgbSwapped()
                    pixmap = QPixmap.fromImage(q_img)
                    scaled = pixmap.scaled(80, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.target_preview.setPixmap(scaled)
                    self.target_preview.setText("")

                # 启用开始按钮
                self.btn_start.setEnabled(True)

    def start_monitoring(self):
        """开始监控"""
        if self.target_feature is None:
            QMessageBox.warning(self, "警告", "请先选择追踪目标！")
            return

        # 创建监控线程
        for idx, (source_path, name) in enumerate(self.video_sources):
            thread = VideoMonitorThread(idx, source_path, name)
            thread.set_target(self.target_feature, threshold=0.5)
            thread.frame_ready.connect(self.on_frame_ready)
            thread.target_found.connect(self.on_target_found)
            thread.running = True
            thread.start()
            self.monitor_threads.append(thread)

        # 更新按钮状态
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_select_target.setEnabled(False)

    def stop_monitoring(self):
        """停止监控"""
        # 停止所有线程
        for thread in self.monitor_threads:
            thread.stop()
            thread.wait()

        self.monitor_threads.clear()

        # 清除告警
        for widget in self.camera_widgets.values():
            widget.clear_alert()

        # 更新按钮状态
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_select_target.setEnabled(True)

    def on_frame_ready(self, camera_idx, frame, detections):
        """更新摄像头画面"""
        if camera_idx in self.camera_widgets:
            # 绘制检测框
            display_frame = frame.copy()
            for det in detections:
                if det['cls_id'] == 0:
                    box = det['box']
                    x1, y1, x2, y2 = box
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            self.camera_widgets[camera_idx].update_frame(display_frame)

    def on_target_found(self, camera_idx, camera_name, frame, detection):
        """发现目标"""
        # 高亮显示摄像头
        if camera_idx in self.camera_widgets:
            self.camera_widgets[camera_idx].show_alert()

            # 在画面上高亮目标
            display_frame = frame.copy()
            box = detection['box']
            x1, y1, x2, y2 = box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(display_frame, "TARGET FOUND!", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            self.camera_widgets[camera_idx].update_frame(display_frame)

        # 添加到告警列表
        similarity = detection.get('similarity', 0)
        timestamp = time.strftime("%H:%M:%S")
        alert_text = f"[{timestamp}] {camera_name} - 相似度: {similarity:.2%}"

        self.alert_list.insertItem(0, alert_text)

        # 保持列表最多50条
        if self.alert_list.count() > 50:
            self.alert_list.takeItem(50)

        # 发送系统通知（可选）
        QMessageBox.warning(
            self,
            f"⚠️ 发现目标 - {camera_name}",
            f"在 {camera_name} 发现目标人物！\n相似度: {similarity:.2%}"
        )

        # 5秒后清除高亮
        QTimer.singleShot(5000, lambda: self.clear_camera_alert(camera_idx))

    def clear_camera_alert(self, camera_idx):
        """清除摄像头告警高亮"""
        if camera_idx in self.camera_widgets:
            self.camera_widgets[camera_idx].clear_alert()

    def closeEvent(self, event):
        """关闭时停止所有监控"""
        self.stop_monitoring()
        event.accept()
