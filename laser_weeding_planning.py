"""
动态激光除草路径规划系统
Dynamic Laser Weeding Path Planning System

该系统模拟移动平台上的激光除草系统，包括：
- 双激光振镜系统（左右各一个）
- 动力学约束（角速度和角加速度）
- 时间窗口约束
- Slack时间硬约束
- 多种路径规划算法
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum
import time


class PlanningAlgorithm(Enum):
    """路径规划算法类型"""
    GREEDY = "greedy"  # 贪心算法
    NEAREST_NEIGHBOR = "nearest_neighbor"  # 最近邻算法
    DYNAMIC_PROGRAMMING = "dynamic_programming"  # 动态规划
    FIFO = "fifo"  # 先进先出


@dataclass
class Weed:
    """杂草对象"""
    x: float  # x坐标 (mm)
    y: float  # y坐标 (mm) 
    width: float  # 宽度 (mm)
    height: float  # 高度 (mm)
    detection_time: float  # 检测到的时间 (s)
    
    def get_center(self) -> Tuple[float, float]:
        """获取杂草中心坐标"""
        return (self.x + self.width / 2, self.y + self.height / 2)
    
    def get_time_window(self, platform_velocity: float, work_zone_start_y: float, 
                       work_zone_end_y: float) -> Tuple[float, float]:
        """计算杂草在工作区的时间窗口"""
        center_y = self.y + self.height / 2
        
        # 计算杂草中心进入和离开工作区的时间
        # 平台向前移动，杂草相对向后移动
        time_enter = self.detection_time + (center_y - work_zone_end_y) / platform_velocity
        time_exit = self.detection_time + (center_y - work_zone_start_y) / platform_velocity
        
        return (time_enter, time_exit)


@dataclass
class GalvanometerConfig:
    """振镜配置参数"""
    max_angular_velocity: float  # 最大角速度 (rad/s)
    max_angular_acceleration: float  # 最大角加速度 (rad/s^2)
    focal_length: float  # 焦距 (mm)
    work_zone_x: Tuple[float, float]  # 工作区x范围 (mm)
    work_zone_y: Tuple[float, float]  # 工作区y范围 (mm)
    
    def position_to_angle(self, x: float, y: float) -> Tuple[float, float]:
        """将位置坐标转换为振镜角度"""
        # 简化模型：假设小角度近似
        theta_x = np.arctan(x / self.focal_length)
        theta_y = np.arctan(y / self.focal_length)
        return (theta_x, theta_y)
    
    def calculate_movement_time(self, from_pos: Tuple[float, float], 
                               to_pos: Tuple[float, float]) -> float:
        """计算振镜从一个位置移动到另一个位置所需的时间"""
        # 转换为角度
        theta1_x, theta1_y = self.position_to_angle(*from_pos)
        theta2_x, theta2_y = self.position_to_angle(*to_pos)
        
        # 计算角度差
        delta_theta_x = abs(theta2_x - theta1_x)
        delta_theta_y = abs(theta2_y - theta1_y)
        
        # 使用梯形速度规划计算时间
        # T = sqrt(4*delta_theta/a) 当不能达到最大速度
        # T = delta_theta/v_max + v_max/a 当能达到最大速度
        
        def calculate_time_single_axis(delta_theta):
            # 达到最大速度所需的角度
            theta_accel = self.max_angular_velocity**2 / self.max_angular_acceleration
            
            if delta_theta <= theta_accel:
                # 不能达到最大速度
                return np.sqrt(4 * delta_theta / self.max_angular_acceleration)
            else:
                # 能达到最大速度
                t_accel = self.max_angular_velocity / self.max_angular_acceleration
                t_const = (delta_theta - theta_accel) / self.max_angular_velocity
                return 2 * t_accel + t_const
        
        # 两个轴的时间取最大值（假设两轴独立运动）
        time_x = calculate_time_single_axis(delta_theta_x)
        time_y = calculate_time_single_axis(delta_theta_y)
        
        return max(time_x, time_y)
    
    def is_in_work_zone(self, x: float, y: float) -> bool:
        """检查位置是否在工作区内"""
        return (self.work_zone_x[0] <= x <= self.work_zone_x[1] and
                self.work_zone_y[0] <= y <= self.work_zone_y[1])


@dataclass
class SystemConfig:
    """系统配置参数"""
    platform_velocity: float  # 平台速度 (mm/s)
    camera_fov_y: float  # 相机视野y方向长度 (mm)
    left_galvo: GalvanometerConfig  # 左振镜
    right_galvo: GalvanometerConfig  # 右振镜
    laser_treatment_time: float  # 单个杂草激光处理时间 (s)
    slack_time: float  # Slack时间（硬约束）(s)
    

class PathPlanner:
    """路径规划器"""
    
    def __init__(self, config: SystemConfig, algorithm: PlanningAlgorithm):
        self.config = config
        self.algorithm = algorithm
        
    def plan(self, weeds: List[Weed], current_time: float, 
            galvo_config: GalvanometerConfig,
            current_position: Optional[Tuple[float, float]] = None) -> List[int]:
        """
        规划激光照射顺序
        
        Args:
            weeds: 待处理的杂草列表
            current_time: 当前时间
            galvo_config: 振镜配置
            current_position: 当前激光位置
            
        Returns:
            杂草处理的索引顺序
        """
        if not weeds:
            return []
        
        if self.algorithm == PlanningAlgorithm.GREEDY:
            return self._greedy_plan(weeds, current_time, galvo_config, current_position)
        elif self.algorithm == PlanningAlgorithm.NEAREST_NEIGHBOR:
            return self._nearest_neighbor_plan(weeds, current_time, galvo_config, current_position)
        elif self.algorithm == PlanningAlgorithm.DYNAMIC_PROGRAMMING:
            return self._dp_plan(weeds, current_time, galvo_config, current_position)
        elif self.algorithm == PlanningAlgorithm.FIFO:
            return self._fifo_plan(weeds, current_time, galvo_config, current_position)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def _check_slack_constraint(self, weed: Weed, arrival_time: float, 
                               galvo_config: GalvanometerConfig) -> bool:
        """检查是否满足slack时间约束（硬约束）"""
        time_enter, time_exit = weed.get_time_window(
            self.config.platform_velocity,
            galvo_config.work_zone_y[0],
            galvo_config.work_zone_y[1]
        )
        
        # 激光到达时间必须在时间窗口内
        if arrival_time < time_enter or arrival_time > time_exit:
            return False
        
        # 完成处理的时间
        finish_time = arrival_time + self.config.laser_treatment_time
        
        # Slack时间 = 离开时间 - 完成时间
        slack = time_exit - finish_time
        
        # 硬约束：slack时间必须大于等于设定的最小slack时间
        return slack >= self.config.slack_time
    
    def _greedy_plan(self, weeds: List[Weed], current_time: float,
                    galvo_config: GalvanometerConfig,
                    current_position: Optional[Tuple[float, float]]) -> List[int]:
        """贪心算法：总是选择剩余时间最少且满足约束的杂草"""
        plan = []
        remaining = set(range(len(weeds)))
        
        # 如果没有当前位置，使用工作区中心
        if current_position is None:
            current_position = (
                (galvo_config.work_zone_x[0] + galvo_config.work_zone_x[1]) / 2,
                (galvo_config.work_zone_y[0] + galvo_config.work_zone_y[1]) / 2
            )
        
        time = current_time
        position = current_position
        
        while remaining:
            best_idx = None
            best_urgency = float('inf')
            best_move_time = 0
            
            for idx in remaining:
                weed = weeds[idx]
                center = weed.get_center()
                
                # 计算移动时间
                move_time = galvo_config.calculate_movement_time(position, center)
                arrival_time = time + move_time
                
                # 检查slack约束
                if not self._check_slack_constraint(weed, arrival_time, galvo_config):
                    continue
                
                # 计算紧急程度（离开时间 - 到达时间）
                time_enter, time_exit = weed.get_time_window(
                    self.config.platform_velocity,
                    galvo_config.work_zone_y[0],
                    galvo_config.work_zone_y[1]
                )
                urgency = time_exit - arrival_time
                
                if urgency < best_urgency:
                    best_urgency = urgency
                    best_idx = idx
                    best_move_time = move_time
            
            if best_idx is None:
                break  # 无法找到满足约束的杂草
            
            plan.append(best_idx)
            remaining.remove(best_idx)
            
            # 更新位置和时间
            position = weeds[best_idx].get_center()
            time += best_move_time + self.config.laser_treatment_time
        
        return plan
    
    def _nearest_neighbor_plan(self, weeds: List[Weed], current_time: float,
                              galvo_config: GalvanometerConfig,
                              current_position: Optional[Tuple[float, float]]) -> List[int]:
        """最近邻算法：总是选择距离最近且满足约束的杂草"""
        plan = []
        remaining = set(range(len(weeds)))
        
        if current_position is None:
            current_position = (
                (galvo_config.work_zone_x[0] + galvo_config.work_zone_x[1]) / 2,
                (galvo_config.work_zone_y[0] + galvo_config.work_zone_y[1]) / 2
            )
        
        time = current_time
        position = current_position
        
        while remaining:
            best_idx = None
            best_distance = float('inf')
            
            for idx in remaining:
                weed = weeds[idx]
                center = weed.get_center()
                
                # 计算移动时间
                move_time = galvo_config.calculate_movement_time(position, center)
                arrival_time = time + move_time
                
                # 检查slack约束
                if not self._check_slack_constraint(weed, arrival_time, galvo_config):
                    continue
                
                # 计算距离
                distance = np.sqrt((center[0] - position[0])**2 + (center[1] - position[1])**2)
                
                if distance < best_distance:
                    best_distance = distance
                    best_idx = idx
            
            if best_idx is None:
                break
            
            plan.append(best_idx)
            remaining.remove(best_idx)
            
            # 更新位置和时间
            weed = weeds[best_idx]
            center = weed.get_center()
            move_time = galvo_config.calculate_movement_time(position, center)
            position = center
            time += move_time + self.config.laser_treatment_time
        
        return plan
    
    def _fifo_plan(self, weeds: List[Weed], current_time: float,
                   galvo_config: GalvanometerConfig,
                   current_position: Optional[Tuple[float, float]]) -> List[int]:
        """先进先出算法：按检测顺序处理满足约束的杂草"""
        plan = []
        
        if current_position is None:
            current_position = (
                (galvo_config.work_zone_x[0] + galvo_config.work_zone_x[1]) / 2,
                (galvo_config.work_zone_y[0] + galvo_config.work_zone_y[1]) / 2
            )
        
        time = current_time
        position = current_position
        
        for idx, weed in enumerate(weeds):
            center = weed.get_center()
            
            # 计算移动时间
            move_time = galvo_config.calculate_movement_time(position, center)
            arrival_time = time + move_time
            
            # 检查slack约束
            if self._check_slack_constraint(weed, arrival_time, galvo_config):
                plan.append(idx)
                position = center
                time += move_time + self.config.laser_treatment_time
        
        return plan
    
    def _dp_plan(self, weeds: List[Weed], current_time: float,
                galvo_config: GalvanometerConfig,
                current_position: Optional[Tuple[float, float]]) -> List[int]:
        """动态规划算法：考虑所有可能的顺序，找到最优解"""
        n = len(weeds)
        if n > 15:  # DP对于大规模问题计算量太大，使用贪心代替
            return self._greedy_plan(weeds, current_time, galvo_config, current_position)
        
        if current_position is None:
            current_position = (
                (galvo_config.work_zone_x[0] + galvo_config.work_zone_x[1]) / 2,
                (galvo_config.work_zone_y[0] + galvo_config.work_zone_y[1]) / 2
            )
        
        # 使用位掩码表示已访问的杂草集合
        # dp[mask][last] = (max_weeds, path)
        dp = {}
        
        def solve(mask, last_idx, position, time):
            if (mask, last_idx) in dp:
                return dp[(mask, last_idx)]
            
            best_count = 0
            best_path = []
            
            for i in range(n):
                if mask & (1 << i):  # 已访问
                    continue
                
                weed = weeds[i]
                center = weed.get_center()
                
                # 计算移动时间
                move_time = galvo_config.calculate_movement_time(position, center)
                arrival_time = time + move_time
                
                # 检查slack约束
                if not self._check_slack_constraint(weed, arrival_time, galvo_config):
                    continue
                
                new_mask = mask | (1 << i)
                new_time = arrival_time + self.config.laser_treatment_time
                
                count, path = solve(new_mask, i, center, new_time)
                count += 1
                
                if count > best_count:
                    best_count = count
                    best_path = [i] + path
            
            dp[(mask, last_idx)] = (best_count, best_path)
            return dp[(mask, last_idx)]
        
        _, path = solve(0, -1, current_position, current_time)
        return path


class LaserWeedingSimulator:
    """激光除草仿真器"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.weeds = []
        self.left_planner = None
        self.right_planner = None
        self.current_time = 0.0
        self.simulation_results = {
            'left': {'treated': [], 'missed': [], 'times': []},
            'right': {'treated': [], 'missed': [], 'times': []}
        }
        
    def set_planning_algorithm(self, algorithm: PlanningAlgorithm):
        """设置路径规划算法"""
        self.left_planner = PathPlanner(self.config, algorithm)
        self.right_planner = PathPlanner(self.config, algorithm)
        
    def add_weeds_from_detection(self, weeds: List[Weed]):
        """从检测结果添加杂草"""
        self.weeds.extend(weeds)
        
    def simulate_step(self, dt: float):
        """执行一步仿真"""
        self.current_time += dt
        
        # 更新每个杂草的位置（相对于平台向后移动）
        for weed in self.weeds:
            weed.y -= self.config.platform_velocity * dt
        
        # 为左右振镜分配杂草并规划路径
        left_weeds = []
        right_weeds = []
        
        for weed in self.weeds:
            center_x, center_y = weed.get_center()
            
            if self.config.left_galvo.is_in_work_zone(center_x, center_y):
                left_weeds.append(weed)
            elif self.config.right_galvo.is_in_work_zone(center_x, center_y):
                right_weeds.append(weed)
        
        # 执行路径规划和处理
        # 这里简化处理，实际应该跟踪每个振镜的状态
        
    def run_simulation(self, total_time: float, dt: float = 0.1):
        """运行完整仿真"""
        steps = int(total_time / dt)
        for _ in range(steps):
            self.simulate_step(dt)
            
    def get_statistics(self):
        """获取仿真统计数据"""
        stats = {}
        for side in ['left', 'right']:
            treated = len(self.simulation_results[side]['treated'])
            missed = len(self.simulation_results[side]['missed'])
            total = treated + missed
            stats[side] = {
                'treated': treated,
                'missed': missed,
                'total': total,
                'success_rate': treated / total if total > 0 else 0
            }
        return stats


def create_default_config() -> SystemConfig:
    """创建默认系统配置"""
    # 振镜参数
    max_angular_velocity = 5.0  # rad/s
    max_angular_acceleration = 50.0  # rad/s^2
    focal_length = 500.0  # mm
    
    # 左振镜工作区 (mm)
    left_galvo = GalvanometerConfig(
        max_angular_velocity=max_angular_velocity,
        max_angular_acceleration=max_angular_acceleration,
        focal_length=focal_length,
        work_zone_x=(-200, 0),
        work_zone_y=(0, 200)
    )
    
    # 右振镜工作区 (mm)
    right_galvo = GalvanometerConfig(
        max_angular_velocity=max_angular_velocity,
        max_angular_acceleration=max_angular_acceleration,
        focal_length=focal_length,
        work_zone_x=(0, 200),
        work_zone_y=(0, 200)
    )
    
    config = SystemConfig(
        platform_velocity=50.0,  # mm/s
        camera_fov_y=500.0,  # mm
        left_galvo=left_galvo,
        right_galvo=right_galvo,
        laser_treatment_time=0.1,  # s
        slack_time=0.05  # s (硬约束)
    )
    
    return config


def generate_random_weeds(n: int, detection_time: float = 0.0) -> List[Weed]:
    """生成随机杂草"""
    weeds = []
    for i in range(n):
        # 交替在左右工作区生成杂草
        if i % 2 == 0:
            # 左工作区
            x = np.random.uniform(-190, -10)
            y = np.random.uniform(10, 190)
        else:
            # 右工作区
            x = np.random.uniform(10, 190)
            y = np.random.uniform(10, 190)
        
        width = np.random.uniform(5, 15)
        height = np.random.uniform(5, 15)
        weeds.append(Weed(x, y, width, height, detection_time))
    return weeds


def visualize_planning(config: SystemConfig, weeds: List[Weed], 
                      algorithm: PlanningAlgorithm, save_path: str = None):
    """可视化路径规划结果"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 绘制左振镜
    ax1.set_title(f'Left Galvanometer - {algorithm.value}')
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_xlim(-250, 50)
    ax1.set_ylim(-50, 250)
    ax1.grid(True)
    
    # 绘制左振镜工作区
    left_zone = patches.Rectangle(
        (config.left_galvo.work_zone_x[0], config.left_galvo.work_zone_y[0]),
        config.left_galvo.work_zone_x[1] - config.left_galvo.work_zone_x[0],
        config.left_galvo.work_zone_y[1] - config.left_galvo.work_zone_y[0],
        linewidth=2, edgecolor='blue', facecolor='lightblue', alpha=0.3
    )
    ax1.add_patch(left_zone)
    
    # 绘制右振镜
    ax2.set_title(f'Right Galvanometer - {algorithm.value}')
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_xlim(-50, 250)
    ax2.set_ylim(-50, 250)
    ax2.grid(True)
    
    # 绘制右振镜工作区
    right_zone = patches.Rectangle(
        (config.right_galvo.work_zone_x[0], config.right_galvo.work_zone_y[0]),
        config.right_galvo.work_zone_x[1] - config.right_galvo.work_zone_x[0],
        config.right_galvo.work_zone_y[1] - config.right_galvo.work_zone_y[0],
        linewidth=2, edgecolor='green', facecolor='lightgreen', alpha=0.3
    )
    ax2.add_patch(right_zone)
    
    # 分配杂草到左右振镜
    left_weeds = []
    right_weeds = []
    left_indices = []
    right_indices = []
    
    for i, weed in enumerate(weeds):
        center_x, center_y = weed.get_center()
        
        if config.left_galvo.is_in_work_zone(center_x, center_y):
            left_weeds.append(weed)
            left_indices.append(i)
        elif config.right_galvo.is_in_work_zone(center_x, center_y):
            right_weeds.append(weed)
            right_indices.append(i)
    
    # 规划路径
    planner = PathPlanner(config, algorithm)
    
    # 左振镜规划
    if left_weeds:
        left_plan = planner.plan(left_weeds, 0.0, config.left_galvo)
        
        # 绘制杂草
        for i, weed in enumerate(left_weeds):
            color = 'red' if i not in left_plan else 'green'
            rect = patches.Rectangle(
                (weed.x, weed.y), weed.width, weed.height,
                linewidth=1, edgecolor=color, facecolor=color, alpha=0.5
            )
            ax1.add_patch(rect)
            ax1.text(weed.x + weed.width/2, weed.y + weed.height/2, 
                    str(i), ha='center', va='center', fontsize=8)
        
        # 绘制路径
        if left_plan:
            path_x = []
            path_y = []
            for idx in left_plan:
                center = left_weeds[idx].get_center()
                path_x.append(center[0])
                path_y.append(center[1])
            ax1.plot(path_x, path_y, 'b-o', linewidth=2, markersize=8, label='Treatment Path')
            
            # 标注顺序
            for order, idx in enumerate(left_plan):
                center = left_weeds[idx].get_center()
                ax1.text(center[0], center[1]-10, f'#{order+1}', 
                        ha='center', va='top', fontsize=10, fontweight='bold', color='blue')
        
        ax1.legend()
        ax1.set_title(f'Left Galvanometer - {algorithm.value}\n'
                     f'Treated: {len(left_plan)}/{len(left_weeds)}')
    
    # 右振镜规划
    if right_weeds:
        right_plan = planner.plan(right_weeds, 0.0, config.right_galvo)
        
        # 绘制杂草
        for i, weed in enumerate(right_weeds):
            color = 'red' if i not in right_plan else 'green'
            rect = patches.Rectangle(
                (weed.x, weed.y), weed.width, weed.height,
                linewidth=1, edgecolor=color, facecolor=color, alpha=0.5
            )
            ax2.add_patch(rect)
            ax2.text(weed.x + weed.width/2, weed.y + weed.height/2, 
                    str(i), ha='center', va='center', fontsize=8)
        
        # 绘制路径
        if right_plan:
            path_x = []
            path_y = []
            for idx in right_plan:
                center = right_weeds[idx].get_center()
                path_x.append(center[0])
                path_y.append(center[1])
            ax2.plot(path_x, path_y, 'g-o', linewidth=2, markersize=8, label='Treatment Path')
            
            # 标注顺序
            for order, idx in enumerate(right_plan):
                center = right_weeds[idx].get_center()
                ax2.text(center[0], center[1]-10, f'#{order+1}', 
                        ha='center', va='top', fontsize=10, fontweight='bold', color='green')
        
        ax2.legend()
        ax2.set_title(f'Right Galvanometer - {algorithm.value}\n'
                     f'Treated: {len(right_plan)}/{len(right_weeds)}')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.savefig('laser_weeding_planning.png', dpi=150, bbox_inches='tight')
        print("Visualization saved to laser_weeding_planning.png")
    
    plt.close()


def compare_algorithms(config: SystemConfig, weeds: List[Weed], save_path: str = None):
    """比较不同算法的性能"""
    algorithms = [
        PlanningAlgorithm.GREEDY,
        PlanningAlgorithm.NEAREST_NEIGHBOR,
        PlanningAlgorithm.FIFO,
        PlanningAlgorithm.DYNAMIC_PROGRAMMING
    ]
    
    results = {}
    
    for algorithm in algorithms:
        planner = PathPlanner(config, algorithm)
        
        # 分配杂草
        left_weeds = []
        right_weeds = []
        
        for weed in weeds:
            center_x, center_y = weed.get_center()
            if config.left_galvo.is_in_work_zone(center_x, center_y):
                left_weeds.append(weed)
            elif config.right_galvo.is_in_work_zone(center_x, center_y):
                right_weeds.append(weed)
        
        # 规划
        left_plan = planner.plan(left_weeds, 0.0, config.left_galvo) if left_weeds else []
        right_plan = planner.plan(right_weeds, 0.0, config.right_galvo) if right_weeds else []
        
        total_weeds = len(left_weeds) + len(right_weeds)
        treated_weeds = len(left_plan) + len(right_plan)
        
        results[algorithm.value] = {
            'total': total_weeds,
            'treated': treated_weeds,
            'missed': total_weeds - treated_weeds,
            'success_rate': treated_weeds / total_weeds if total_weeds > 0 else 0
        }
    
    # 打印结果
    print("\n" + "="*60)
    print("Algorithm Performance Comparison")
    print("="*60)
    print(f"{'Algorithm':<20} {'Total':<8} {'Treated':<10} {'Missed':<8} {'Success Rate':<12}")
    print("-"*60)
    
    for algo, result in results.items():
        print(f"{algo:<20} {result['total']:<8} {result['treated']:<10} "
              f"{result['missed']:<8} {result['success_rate']:.2%}")
    
    print("="*60)
    
    # 可视化比较
    fig, ax = plt.subplots(figsize=(12, 6))
    
    algo_names = list(results.keys())
    success_rates = [results[algo]['success_rate'] * 100 for algo in algo_names]
    
    bars = ax.bar(algo_names, success_rates, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
    
    ax.set_xlabel('Algorithm', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title(f'Algorithm Performance Comparison (Slack Time = {config.slack_time}s)', fontsize=14)
    ax.set_ylim(0, 105)
    ax.grid(True, axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison chart saved to {save_path}")
    else:
        plt.savefig('algorithm_comparison.png', dpi=150, bbox_inches='tight')
        print("Comparison chart saved to algorithm_comparison.png")
    
    plt.close()
    
    return results


def main():
    """主函数"""
    print("="*60)
    print("Dynamic Laser Weeding Path Planning System")
    print("动态激光除草路径规划系统")
    print("="*60)
    
    # 创建系统配置
    config = create_default_config()
    
    print(f"\nSystem Configuration:")
    print(f"  Platform Velocity: {config.platform_velocity} mm/s")
    print(f"  Laser Treatment Time: {config.laser_treatment_time} s")
    print(f"  Slack Time Constraint: {config.slack_time} s (Hard Constraint)")
    print(f"  Left Galvo Work Zone: X{config.left_galvo.work_zone_x}, Y{config.left_galvo.work_zone_y}")
    print(f"  Right Galvo Work Zone: X{config.right_galvo.work_zone_x}, Y{config.right_galvo.work_zone_y}")
    
    # 生成测试杂草
    np.random.seed(42)
    weeds = generate_random_weeds(30)
    
    print(f"\nGenerated {len(weeds)} random weeds for testing")
    
    # 测试单个算法可视化
    print("\nGenerating visualizations for each algorithm...")
    for algorithm in PlanningAlgorithm:
        visualize_planning(config, weeds, algorithm, 
                         f'planning_{algorithm.value}.png')
    
    # 比较算法性能
    print("\nComparing algorithm performance...")
    comparison_results = compare_algorithms(config, weeds, 'algorithm_comparison.png')
    
    # 测试不同slack时间的影响
    print("\n" + "="*60)
    print("Testing Different Slack Time Constraints")
    print("="*60)
    
    slack_times = [0.0, 0.02, 0.05, 0.1, 0.2]
    algorithm = PlanningAlgorithm.GREEDY
    
    slack_results = []
    for slack in slack_times:
        config.slack_time = slack
        planner = PathPlanner(config, algorithm)
        
        left_weeds = []
        right_weeds = []
        
        for weed in weeds:
            center_x, center_y = weed.get_center()
            if config.left_galvo.is_in_work_zone(center_x, center_y):
                left_weeds.append(weed)
            elif config.right_galvo.is_in_work_zone(center_x, center_y):
                right_weeds.append(weed)
        
        left_plan = planner.plan(left_weeds, 0.0, config.left_galvo) if left_weeds else []
        right_plan = planner.plan(right_weeds, 0.0, config.right_galvo) if right_weeds else []
        
        total = len(left_weeds) + len(right_weeds)
        treated = len(left_plan) + len(right_plan)
        
        success_rate = treated / total if total > 0 else 0
        slack_results.append({
            'slack': slack,
            'success_rate': success_rate
        })
        
        print(f"Slack Time: {slack:.2f}s -> Success Rate: {success_rate:.2%} ({treated}/{total})")
    
    # 绘制slack时间影响图
    fig, ax = plt.subplots(figsize=(10, 6))
    
    slack_values = [r['slack'] for r in slack_results]
    rates = [r['success_rate'] * 100 for r in slack_results]
    
    ax.plot(slack_values, rates, 'b-o', linewidth=2, markersize=8)
    ax.set_xlabel('Slack Time Constraint (s)', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title(f'Impact of Slack Time Constraint on Success Rate ({algorithm.value})', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    
    for x, y in zip(slack_values, rates):
        ax.text(x, y+2, f'{y:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('slack_time_impact.png', dpi=150, bbox_inches='tight')
    print("\nSlack time impact chart saved to slack_time_impact.png")
    plt.close()
    
    print("\n" + "="*60)
    print("Simulation Complete!")
    print("Generated files:")
    print("  - planning_greedy.png")
    print("  - planning_nearest_neighbor.png")
    print("  - planning_fifo.png")
    print("  - planning_dynamic_programming.png")
    print("  - algorithm_comparison.png")
    print("  - slack_time_impact.png")
    print("="*60)


if __name__ == "__main__":
    main()
