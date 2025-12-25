# 跨镜头追踪系统 (Multi-Camera Cross-View Tracking)

## 功能说明

这个系统可以追踪同一个人在不同摄像头之间的移动，即使摄像头之间没有视野重叠。

### 核心功能

1. **跨镜头人物识别** - 使用ReID（Re-Identification）技术识别同一个人
2. **全局ID管理** - 为每个人分配唯一的全局ID，即使在不同摄像头之间
3. **外观特征匹配** - 基于衣服颜色、纹理等特征进行匹配
4. **多摄像头并行处理** - 同时处理多个视频流

## 系统架构

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Camera 1   │     │  Camera 2   │     │  Camera 3   │
│  (cam_0)    │     │  (cam_1)    │     │  (cam_2)    │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │
       │   Local Tracking  │                   │
       │   (BoT-SORT)      │                   │
       │                   │                   │
       └───────────┬───────┴───────────────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │ Multi-Camera Tracker │
        │  (Global ID Manager) │
        └──────────────────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  ReID Feature        │
        │  Matching            │
        └──────────────────────┘
```

## 文件说明

### 核心模块

1. **`src/core/multi_camera_tracker.py`**
   - 全局ID管理器
   - 跨镜头特征匹配逻辑
   - 维护全局人物数据库

2. **`src/core/reid_extractor.py`**
   - ReID特征提取
   - 外观相似度计算
   - 支持颜色、纹理等特征

3. **`src/core/detection.py`** (已更新)
   - 单镜头检测和追踪
   - 集成跨镜头追踪功能
   - 显示全局ID和本地ID

4. **`trackers/botsort_reid.yaml`**
   - BoT-SORT配置，启用ReID
   - 优化的追踪参数

### 示例代码

- **`example_multi_camera.py`** - 多镜头系统使用示例

## 使用方法

### 方法1：单独使用跨镜头追踪器

```python
from src.core.multi_camera_tracker import MultiCameraTracker
from src.core.detection import VideoDetector

# 创建全局追踪器
multi_cam_tracker = MultiCameraTracker(
    similarity_threshold=0.6,  # ReID相似度阈值
    max_time_gap=30            # 最大时间间隔（秒）
)

# 为每个摄像头创建检测器
detector1 = VideoDetector(
    camera_id="cam_0",
    multi_cam_tracker=multi_cam_tracker
)

detector2 = VideoDetector(
    camera_id="cam_1",
    multi_cam_tracker=multi_cam_tracker
)

# 处理帧
detections1, _ = detector1.detect(frame1, tracking_enabled=True)
detections2, _ = detector2.detect(frame2, tracking_enabled=True)

# 检查全局ID
for det in detections1:
    local_id = det['id']
    global_id = det['global_id']
    print(f"Camera 1: Local ID {local_id} → Global ID {global_id}")
```

### 方法2：使用多镜头系统类

```python
from example_multi_camera import MultiCameraSystem

# 定义视频源
video_sources = [
    "videos/entrance.mp4",
    "videos/corridor.mp4",
    "videos/exit.mp4"
]

# 创建系统
system = MultiCameraSystem(video_sources)

# 运行
system.run()
```

## 参数调优

### ReID相似度阈值 (similarity_threshold)

- **范围**: 0.0 - 1.0
- **默认**: 0.6
- **说明**:
  - 越高：匹配越严格，减少错误匹配，但可能漏掉真实匹配
  - 越低：匹配越宽松，增加匹配率，但可能产生错误匹配

**建议值**:
- 摄像头环境相似（室内）: 0.5 - 0.6
- 摄像头环境差异大（室内+室外）: 0.4 - 0.5
- 光线条件好，画质清晰: 0.6 - 0.7

### 最大时间间隔 (max_time_gap)

- **范围**: 1 - 300 秒
- **默认**: 30 秒
- **说明**: 两次出现之间的最大时间间隔

**建议值**:
- 摄像头距离近（5-10米）: 10-15 秒
- 摄像头距离中等（10-50米）: 20-30 秒
- 摄像头距离远（50米+）: 30-60 秒

## ID显示格式

在画面中，人物标签会显示：

- **单摄像头模式**: `ID: 5 (Blue)` - 本地ID + 颜色
- **跨镜头模式**: `G12 (L5) (Blue)` - G12=全局ID, L5=本地ID

## 性能考虑

### 计算开销

跨镜头追踪会增加额外计算：

1. **ReID特征提取**: 每个人每帧提取一次 (~5ms/人)
2. **特征匹配**: 与全局数据库匹配 (~1-2ms/人)

**优化建议**:
- 限制全局数据库大小（保留最近100人）
- 定期清理超时的旧ID
- 考虑每N帧提取一次特征（而不是每帧）

### 摄像头数量

- **2-3个摄像头**: 流畅运行
- **4-6个摄像头**: 建议使用GPU
- **6个以上**: 建议分布式部署

## 故障排除

### 问题1: 相同人物被分配不同的全局ID

**原因**:
- ReID相似度阈值太高
- 光线、角度差异太大
- 时间间隔超过max_time_gap

**解决**:
- 降低similarity_threshold到0.4-0.5
- 增加max_time_gap
- 检查摄像头光线条件

### 问题2: 不同人物被分配相同的全局ID

**原因**:
- ReID相似度阈值太低
- 人物外观非常相似（如制服）

**解决**:
- 提高similarity_threshold到0.6-0.7
- 考虑使用更强的ReID模型（FastReID）

### 问题3: 系统运行很慢

**原因**:
- 摄像头数量太多
- 全局数据库太大

**解决**:
- 降低视频分辨率
- 启用cleanup_old_tracks()定期清理
- 减少pose检测频率

## 未来改进

### 计划中的功能

1. **集成FastReID模型**
   - 更准确的ReID特征
   - 专门训练的行人重识别模型

2. **空间拓扑关系**
   - 摄像头之间的物理关系
   - 基于位置的匹配优化

3. **轨迹预测**
   - 预测人物下一步出现的摄像头
   - 加快匹配速度

4. **Web界面**
   - 实时监控所有摄像头
   - 可视化全局ID轨迹

## 示例场景

### 场景1: 商场监控

```
入口摄像头 → 大厅摄像头 → 楼梯摄像头 → 出口摄像头
   cam_0        cam_1          cam_2          cam_3
```

**用途**: 追踪顾客流动路径，分析热门区域

### 场景2: 安防监控

```
小区门口 → 楼栋入口 → 电梯厅 → 楼层走廊
 cam_0       cam_1       cam_2      cam_3
```

**用途**: 追踪可疑人员，记录活动轨迹

### 场景3: 工厂监控

```
入口门禁 → 生产区A → 生产区B → 仓库区
 cam_0       cam_1       cam_2      cam_3
```

**用途**: 员工活动追踪，考勤管理

## 技术支持

如有问题，请查看：
1. 日志输出中的错误信息
2. 调整参数后重试
3. 确认视频源可访问

## 更新日志

- **2025-12-25**: 初版发布
  - 基础跨镜头追踪功能
  - BoT-SORT ReID集成
  - 手工特征提取器
