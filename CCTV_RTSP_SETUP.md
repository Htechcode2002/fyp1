# CCTV/RTSP Camera Setup Guide

## 新功能：CCTV/RTSP 摄像头连接

现在可以直接在 Video Sources Management 页面配置 CCTV 摄像头，无需手动构建 RTSP URL！

## 使用方法

### 1. 添加新的视频源
在 Config 页面，点击 **"+ Add New Source"**

### 2. 选择 CCTV/RTSP 类型
在视频源卡片中，选择 **"CCTV/RTSP"** 单选按钮

### 3. 填写 CCTV 配置信息

系统会显示一个蓝色的配置表单，填写以下信息：

- **IP Address**: 摄像头的 IP 地址（例如：`192.168.0.79`）
- **Username**: 登录用户名（例如：`admin`）
- **Password**: 登录密码（密码会被隐藏显示）
- **Channel**: 通道号（通常是 `1`）
- **Subtype**:
  - `Main Stream (High Quality)` - 主码流（高清，占用带宽大）
  - `Sub Stream (Recommended)` - 子码流（推荐，占用带宽小）

### 4. 实时预览 RTSP URL

当你输入配置信息时，系统会自动生成 RTSP URL 并显示在表单底部。
- 密码会被 `****` 遮盖
- 鼠标悬停在 URL 上可以看到完整的 URL（包含密码）

### 5. 保存配置

点击 **"Save Changes"** 按钮保存配置。

系统会自动：
- 构建完整的 RTSP URL：`rtsp://username:password@ip:554/cam/realmonitor?channel=1&subtype=1`
- 保存 CCTV 配置信息到 JSON 文件
- 下次编辑时会自动填充之前的配置

## 你的摄像头配置示例

根据你提供的资料，配置如下：

```
IP Address: 192.168.0.79
Username: admin
Password: Charles_20022002
Channel: 1
Subtype: Sub Stream (Recommended)
```

系统生成的 RTSP URL：
```
rtsp://admin:Charles_20022002@192.168.0.79:554/cam/realmonitor?channel=1&subtype=1
```

## 特点

✅ **自动生成 URL** - 无需手动构建复杂的 RTSP URL
✅ **密码保护** - 密码在界面上被遮盖，只在需要时显示
✅ **配置保存** - 配置信息会被保存，方便下次修改
✅ **实时预览** - 输入时立即看到生成的 URL
✅ **简单易用** - 只需填写几个字段即可完成配置

## 技术说明

### RTSP URL 格式
```
rtsp://{username}:{password}@{ip}:554/cam/realmonitor?channel={channel}&subtype={subtype}
```

### Subtype 参数
- `0` = 主码流（高清，1080p/4K，占用带宽大）
- `1` = 子码流（标清，适合远程监控，推荐使用）

### 保存的数据结构

配置会以以下格式保存到 `config.json`：

```json
{
  "id": "3b14de4c",
  "name": "Front Gate",
  "type": "cctv",
  "path": "rtsp://admin:Charles_20022002@192.168.0.79:554/cam/realmonitor?channel=1&subtype=1",
  "location": "Front Gate",
  "danger_threshold": 100,
  "loitering_threshold": 5,
  "fall_threshold": 2,
  "rtsp_config": {
    "ip": "192.168.0.79",
    "username": "admin",
    "password": "Charles_20022002",
    "channel": "1",
    "subtype": 1
  }
}
```

## 常见问题

### Q: 为什么推荐使用子码流？
A: 子码流（Subtype=1）占用带宽更小，适合远程监控和长时间运行。主码流虽然画质更高，但会消耗更多网络和系统资源。

### Q: 可以同时连接多个 CCTV 摄像头吗？
A: 可以！只需要点击 "+ Add New Source" 添加多个视频源，每个都可以配置不同的 CCTV 摄像头。

### Q: 配置保存后可以修改吗？
A: 可以！系统会保存所有配置信息（包括 IP、用户名、密码等），下次编辑时会自动填充，方便修改。

### Q: RTSP URL 什么时候生成？
A: URL 会在你输入配置信息时实时生成并显示预览，最终在点击 "Save Changes" 时保存。

## 更新日志

**Version: 2026-01-08**
- ✅ 添加 CCTV/RTSP 单选按钮
- ✅ 创建 RTSP 配置表单（IP、用户名、密码、通道、码流类型）
- ✅ 实时 RTSP URL 生成和预览
- ✅ 密码遮盖保护
- ✅ 配置保存和加载功能
- ✅ 支持多个 CCTV 摄像头
