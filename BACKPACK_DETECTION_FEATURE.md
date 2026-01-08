# Backpack Detection Feature

## 📦 功能概述

在数据库中添加了 **backpack**（背包）检测字段，用于记录行人是否携带背包。

## ✅ 已完成的修改

### 1. 数据库层 (`src/core/database.py`)

#### 表结构更新
```sql
CREATE TABLE IF NOT EXISTS crossing_events (
    ...
    handbag TINYINT DEFAULT 0,
    backpack TINYINT DEFAULT 0,  -- 新增字段
    timestamp DATETIME
)
```

#### 自动迁移
- 添加自动检测和创建 `backpack` 列的逻辑
- 兼容旧数据库，自动升级表结构

#### API 更新
```python
def insert_event(..., handbag=0, backpack=0):
    # 新增 backpack 参数
```

### 2. 检测逻辑 (`src/core/detection.py`)

#### 缓存初始化
```python
# Backpack Detection Tracking (1-confirmation caching)
self.backpack_cache = {}  # track_id -> 1 if has backpack, 0 if no backpack
self.backpack_confirmed = {}  # track_id -> bool (True if confirmed)
```

#### YOLO 类别
- **Class 24**: Backpack (背包)
- **Class 26**: Handbag (手提包)

#### 检测算法
```python
# === BACKPACK DETECTION ===
# 1. 收集所有未确认的人员检测
# 2. 收集所有背包检测 (Class 24)
# 3. 距离匹配算法：
#    - 计算人员和背包的欧氏距离
#    - 阈值：人员身高的 80%
#    - 如果距离 < 阈值 → has_backpack = 1
# 4. 1-confirmation 缓存 (只检测一次)
```

#### 标签显示
- 检测到背包时，在人头上方显示 `[BP]` 标签
- 颜色编码：绿色 (#10b981)

#### 数据库记录
```python
self.db.insert_event(
    ...
    handbag=handbag_val,
    backpack=backpack_val  # 新增
)
```

#### 缓存清理
- 在 `reset_analytics()` 中清理 backpack 缓存
- 防止内存泄漏

### 3. UI 界面 (`src/ui/data_view_page.py`)

#### 表格列更新
- 列数：12 → 13
- 新增列：**Backpack** (第 11 列)
- 列顺序：ID, Time, Location, Line, Left, Right, Color, Gender, Age, Mask, Handbag, **Backpack**, Video ID

#### 过滤器
```python
# 新增 Backpack 过滤器 (Row 4)
self.combo_backpack = QComboBox()
self.combo_backpack.addItems(["All", "With Backpack", "No Backpack"])
```

#### 数据显示
- 有背包：`🎒 Yes` (绿色粗体)
- 无背包：`—` (灰色)

## 🎯 使用方法

### 检测流程

1. **系统自动检测**
   - 当人员进入画面时，系统自动检测附近的背包 (Class 24)
   - 使用距离匹配算法判断背包是否属于该人员

2. **1-Confirmation 缓存**
   - 每个 track_id 只检测一次
   - 结果永久缓存，不再重复检测
   - 提高性能，减少资源消耗

3. **数据记录**
   - 当人员跨越计数线时
   - 背包状态自动保存到数据库
   - `backpack = 1` (有背包) 或 `backpack = 0` (无背包)

### 数据查询

#### 在 Database 页面查看：

1. **过滤背包记录**
   - Backpack 下拉框选择：
     - "All" - 显示所有记录
     - "With Backpack" - 只显示有背包的记录
     - "No Backpack" - 只显示无背包的记录

2. **表格显示**
   - Backpack 列显示：
     - `🎒 Yes` - 有背包 (绿色)
     - `—` - 无背包

3. **组合过滤**
   - 可以结合其他过滤器：
     - 时间范围
     - Gender (性别)
     - Handbag (手提包)
     - Backpack (背包)
     - Mask (口罩)

## 📊 数据库查询示例

### 查询所有有背包的记录
```sql
SELECT * FROM crossing_events WHERE backpack = 1;
```

### 统计背包携带率
```sql
SELECT
    COUNT(*) as total,
    SUM(backpack) as with_backpack,
    ROUND(SUM(backpack) * 100.0 / COUNT(*), 2) as percentage
FROM crossing_events
WHERE timestamp >= NOW() - INTERVAL 1 HOUR;
```

### 按性别统计背包
```sql
SELECT
    gender,
    COUNT(*) as total,
    SUM(backpack) as with_backpack
FROM crossing_events
WHERE gender IS NOT NULL
GROUP BY gender;
```

### 同时携带手提包和背包的人数
```sql
SELECT COUNT(*)
FROM crossing_events
WHERE handbag = 1 AND backpack = 1;
```

## 🔧 技术细节

### YOLO COCO Classes
```
Class 0:  Person (人)
Class 24: Backpack (背包) ✅
Class 26: Handbag (手提包) ✅
```

### 距离匹配算法
```python
# 计算人员和背包的距离
distance = sqrt((px - bx)² + (py - by)²)

# 动态阈值 (基于人员身高)
person_height = p_y2 - p_y1
max_distance = person_height * 0.8  # 80% 的人员身高

# 判断
if distance < max_distance:
    has_backpack = 1
```

**为什么用 80%？**
- 背包通常在人员肩部或背部
- 距离不会超过人员身高的大部分
- 80% 是一个经验值，平衡准确率和召回率

### 1-Confirmation 缓存机制

**传统方式**（浪费资源）：
```
帧 1: 检测 → 有背包
帧 2: 检测 → 有背包
帧 3: 检测 → 有背包
...
帧 100: 检测 → 有背包  ❌ 浪费 99 次检测
```

**1-Confirmation**（高效）：
```
帧 1: 检测 → 有背包 → 缓存 ✅
帧 2: 跳过检测 → 使用缓存
帧 3: 跳过检测 → 使用缓存
...
帧 100: 跳过检测 → 使用缓存  ✅ 节省 99 次检测
```

### 内存优化
```python
# 缓存大小监控（每 300 帧）
cache_sizes = {
    'backpack_cache': len(self.backpack_cache),
    ...
}

# 视频循环时清理
def reset_analytics(self):
    self.backpack_cache.clear()
    self.backpack_confirmed.clear()
```

## 🎨 UI 设计

### 标签显示
```
人员头顶标签格式：
[ID] [Color] [Gender Age] [MASK] [BAG] [BP]

示例：
#42 Blue M 25 [NO MASK] [BAG] [BP]
```

### 表格样式
| Backpack | 颜色 | 字体 |
|----------|------|------|
| 🎒 Yes | #10b981 (绿色) | Arial 10 Bold |
| — | #94a3b8 (灰色) | Arial 10 Normal |

### 过滤器布局
```
Row 1: [Gender]  [Color]   [Mask]
Row 2: [Handbag] [Limit]
Row 3: [Backpack]  ← 新增
```

## 🔄 与 Handbag 的对比

| 特性 | Handbag (手提包) | Backpack (背包) |
|------|-----------------|----------------|
| YOLO Class | 26 | 24 |
| 数据库字段 | handbag | backpack |
| 标签显示 | [BAG] | [BP] |
| UI 颜色 | 紫色 (#8b5cf6) | 绿色 (#10b981) |
| 表情符号 | 👜 | 🎒 |
| 检测算法 | 距离匹配 (80%) | 距离匹配 (80%) |
| 缓存机制 | 1-Confirmation | 1-Confirmation |

## 📝 版本历史

**Version: 2026-01-08**
- ✅ 添加 `backpack` 数据库字段
- ✅ 实现 Class 24 (Backpack) 检测
- ✅ 1-Confirmation 缓存机制
- ✅ 距离匹配算法 (80% 阈值)
- ✅ 标签显示 [BP]
- ✅ UI 过滤器和表格列
- ✅ 数据库自动迁移
- ✅ 缓存清理和内存优化

## 🚀 后续优化建议

### 1. 调整距离阈值
如果检测不准确，可以调整：
```python
# detection.py 第 841 行
max_distance = person_height * 0.8  # 可调整为 0.6-1.0
```

### 2. 背包类型细分
未来可以扩展为：
- 大背包 (Hiking backpack)
- 小背包 (School backpack)
- 行李箱 (Suitcase)

### 3. 组合分析
```sql
-- 携带物品统计
SELECT
    CASE
        WHEN handbag = 0 AND backpack = 0 THEN 'No bag'
        WHEN handbag = 1 AND backpack = 0 THEN 'Handbag only'
        WHEN handbag = 0 AND backpack = 1 THEN 'Backpack only'
        WHEN handbag = 1 AND backpack = 1 THEN 'Both'
    END as bag_type,
    COUNT(*) as count
FROM crossing_events
GROUP BY bag_type;
```

## 💡 使用提示

1. **检测准确率**
   - 背包必须在画面中清晰可见
   - 距离太远或遮挡严重会影响检测
   - 子码流画质可能影响小物体检测

2. **性能考虑**
   - 1-Confirmation 大幅减少计算量
   - 适合实时流监控
   - 不影响其他检测功能

3. **数据分析**
   - 可用于校园/公司安全监控
   - 分析人流携带物品习惯
   - 结合其他属性做行为分析

## 🎯 总结

背包检测功能已完全集成到系统中，使用与 Handbag 相同的技术架构：
- ✅ 自动检测（Class 24）
- ✅ 1-Confirmation 缓存
- ✅ 数据库存储
- ✅ UI 显示和过滤
- ✅ 高性能、低资源消耗

现在系统可以同时检测和记录：
- 👜 Handbag (手提包)
- 🎒 Backpack (背包)

享受更强大的行人分析能力！🚀
