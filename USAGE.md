# 使用指南 (User Guide)

## 快速开始 (Quick Start)

### 1. 安装依赖 (Install Dependencies)

```bash
pip install -r requirements.txt
```

### 2. 运行仿真 (Run Simulation)

```bash
python laser_weeding_planning.py
```

运行后会生成6个PNG图片文件，展示不同算法的规划结果和性能对比。

## 核心概念 (Core Concepts)

### Slack时间硬约束 (Slack Time Hard Constraint)

Slack时间是一个关键的安全约束，定义为：

```
slack_time = time_exit - treatment_finish_time
```

其中：
- `time_exit`: 杂草离开工作区的时间
- `treatment_finish_time`: 激光处理完成的时间

**硬约束要求**: `slack_time >= config.slack_time`

这确保了激光处理有足够的安全余量，防止处理到一半杂草就离开了工作区。

### 时间窗口 (Time Window)

每个杂草在工作区内的可处理时间窗口为：

```python
time_enter = detection_time + (weed_y - work_zone_end_y) / platform_velocity
time_exit = detection_time + (weed_y - work_zone_start_y) / platform_velocity
```

激光必须在 `[time_enter, time_exit]` 时间窗口内完成处理。

### 振镜动力学 (Galvanometer Dynamics)

振镜移动时间考虑了角速度和角加速度约束，使用梯形速度规划：

- 如果移动角度较小，无法达到最大角速度
- 如果移动角度较大，会经历加速、匀速、减速三个阶段

## 自定义配置 (Custom Configuration)

### 修改系统参数

在 `laser_weeding_planning.py` 中找到 `create_default_config()` 函数：

```python
def create_default_config() -> SystemConfig:
    # 修改这些参数
    max_angular_velocity = 5.0  # rad/s
    max_angular_acceleration = 50.0  # rad/s^2
    focal_length = 500.0  # mm
    
    config = SystemConfig(
        platform_velocity=50.0,  # mm/s
        laser_treatment_time=0.1,  # s
        slack_time=0.05  # s - 修改这个值来测试不同的slack约束
    )
    
    return config
```

### 修改工作区范围

```python
# 左振镜工作区
left_galvo = GalvanometerConfig(
    work_zone_x=(-200, 0),  # 修改x范围
    work_zone_y=(0, 200),   # 修改y范围
    # ... 其他参数
)

# 右振镜工作区
right_galvo = GalvanometerConfig(
    work_zone_x=(0, 200),   # 修改x范围
    work_zone_y=(0, 200),   # 修改y范围
    # ... 其他参数
)
```

### 修改测试场景

生成不同数量的杂草：

```python
# 在 main() 函数中
weeds = generate_random_weeds(30)  # 修改数量
```

或使用自定义杂草数据：

```python
# 手动创建杂草
custom_weeds = [
    Weed(x=-100, y=50, width=10, height=10, detection_time=0.0),
    Weed(x=-50, y=80, width=8, height=12, detection_time=0.0),
    # ... 更多杂草
]
```

## 添加新的路径规划算法

### 步骤1: 在PlanningAlgorithm枚举中添加新算法

```python
class PlanningAlgorithm(Enum):
    GREEDY = "greedy"
    NEAREST_NEIGHBOR = "nearest_neighbor"
    DYNAMIC_PROGRAMMING = "dynamic_programming"
    FIFO = "fifo"
    YOUR_ALGORITHM = "your_algorithm"  # 添加新算法
```

### 步骤2: 在PathPlanner类中实现算法

```python
def _your_algorithm_plan(self, weeds: List[Weed], current_time: float,
                        galvo_config: GalvanometerConfig,
                        current_position: Optional[Tuple[float, float]]) -> List[int]:
    """你的算法实现"""
    plan = []
    
    # 实现你的规划逻辑
    # 必须调用 self._check_slack_constraint() 检查约束
    
    for idx, weed in enumerate(weeds):
        center = weed.get_center()
        move_time = galvo_config.calculate_movement_time(current_position, center)
        arrival_time = current_time + move_time
        
        # 检查slack约束（硬约束）
        if self._check_slack_constraint(weed, arrival_time, galvo_config):
            plan.append(idx)
            current_position = center
            current_time = arrival_time + self.config.laser_treatment_time
    
    return plan
```

### 步骤3: 在plan()方法中添加调用

```python
def plan(self, weeds: List[Weed], current_time: float, 
        galvo_config: GalvanometerConfig,
        current_position: Optional[Tuple[float, float]] = None) -> List[int]:
    # ... 现有代码 ...
    elif self.algorithm == PlanningAlgorithm.YOUR_ALGORITHM:
        return self._your_algorithm_plan(weeds, current_time, galvo_config, current_position)
```

### 步骤4: 测试新算法

```python
# 在main()函数中添加
visualize_planning(config, weeds, PlanningAlgorithm.YOUR_ALGORITHM, 
                  'planning_your_algorithm.png')
```

## 性能指标 (Performance Metrics)

系统计算以下指标：

- **Total**: 总杂草数量
- **Treated**: 成功处理的杂草数量
- **Missed**: 未能处理的杂草数量（由于约束限制）
- **Success Rate**: 成功率 = Treated / Total

## 常见问题 (FAQ)

### Q1: 为什么某些算法成功率很低？

A: 这可能是由于：
1. Slack时间约束太严格
2. 杂草分布太密集
3. 振镜移动速度受限

尝试：
- 减小 `config.slack_time`
- 增大 `max_angular_velocity` 或 `max_angular_acceleration`
- 调整杂草分布

### Q2: 如何增加工作区大小？

A: 修改 `GalvanometerConfig` 中的 `work_zone_x` 和 `work_zone_y` 参数。
注意：工作区越大，振镜移动时间可能越长。

### Q3: 动态规划算法为什么对大规模问题不适用？

A: 动态规划的时间复杂度是 O(n * 2^n)，对于超过15个杂草的情况，
计算时间会指数级增长。代码会自动降级到贪心算法。

### Q4: 如何使用实际的杂草检测数据？

A: 将检测结果转换为 `Weed` 对象列表：

```python
import json

# 假设你有JSON格式的检测结果
with open('detection_results.json', 'r') as f:
    detections = json.load(f)

weeds = []
for det in detections:
    weed = Weed(
        x=det['bbox']['x'],
        y=det['bbox']['y'],
        width=det['bbox']['width'],
        height=det['bbox']['height'],
        detection_time=det['timestamp']
    )
    weeds.append(weed)
```

## 扩展方向 (Future Extensions)

1. **实时仿真**: 添加动画展示平台移动和激光处理过程
2. **多目标优化**: 同时考虑成功率、能耗、时间等多个目标
3. **自适应参数**: 根据杂草分布自动调整slack时间
4. **机器学习**: 使用强化学习训练路径规划策略
5. **硬件集成**: 连接实际的振镜控制器和相机系统

## 技术支持 (Support)

如有问题或建议，请在GitHub仓库提交Issue。
