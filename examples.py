"""
示例脚本：展示如何使用路径规划系统
Example script demonstrating how to use the path planning system
"""

from laser_weeding_planning import (
    create_default_config,
    generate_random_weeds,
    visualize_planning,
    compare_algorithms,
    PlanningAlgorithm,
    PathPlanner,
    Weed
)
import numpy as np


def example_1_basic_usage():
    """示例1：基本使用"""
    print("\n" + "="*60)
    print("Example 1: Basic Usage")
    print("="*60)
    
    # 创建系统配置
    config = create_default_config()
    
    # 生成随机杂草
    np.random.seed(42)
    weeds = generate_random_weeds(20)
    
    # 使用贪心算法进行路径规划
    visualize_planning(config, weeds, PlanningAlgorithm.GREEDY, 
                      'example_1_greedy.png')
    
    print(f"Generated {len(weeds)} weeds")
    print("Visualization saved to example_1_greedy.png")


def example_2_compare_algorithms():
    """示例2：比较不同算法"""
    print("\n" + "="*60)
    print("Example 2: Compare Algorithms")
    print("="*60)
    
    config = create_default_config()
    np.random.seed(42)
    weeds = generate_random_weeds(25)
    
    # 比较所有算法
    results = compare_algorithms(config, weeds, 'example_2_comparison.png')
    
    print("\nResults:")
    for algo, result in results.items():
        print(f"  {algo}: {result['success_rate']:.1%} success rate")


def example_3_custom_weeds():
    """示例3：使用自定义杂草数据"""
    print("\n" + "="*60)
    print("Example 3: Custom Weed Data")
    print("="*60)
    
    config = create_default_config()
    
    # 手动创建杂草（模拟检测结果）
    custom_weeds = [
        # 左工作区的杂草
        Weed(x=-150, y=50, width=10, height=10, detection_time=0.0),
        Weed(x=-100, y=80, width=8, height=12, detection_time=0.0),
        Weed(x=-80, y=120, width=12, height=8, detection_time=0.0),
        Weed(x=-120, y=150, width=10, height=10, detection_time=0.0),
        
        # 右工作区的杂草
        Weed(x=50, y=60, width=9, height=11, detection_time=0.0),
        Weed(x=100, y=90, width=11, height=9, detection_time=0.0),
        Weed(x=150, y=130, width=10, height=10, detection_time=0.0),
    ]
    
    print(f"Created {len(custom_weeds)} custom weeds")
    
    # 测试不同算法
    for algorithm in [PlanningAlgorithm.GREEDY, PlanningAlgorithm.NEAREST_NEIGHBOR]:
        visualize_planning(config, custom_weeds, algorithm,
                         f'example_3_{algorithm.value}.png')
        print(f"  Visualization for {algorithm.value} saved")


def example_4_test_slack_times():
    """示例4：测试不同的slack时间约束"""
    print("\n" + "="*60)
    print("Example 4: Test Different Slack Times")
    print("="*60)
    
    config = create_default_config()
    np.random.seed(42)
    weeds = generate_random_weeds(30)
    
    slack_times = [0.0, 0.05, 0.1, 0.15, 0.2]
    
    print(f"\nTesting with {len(weeds)} weeds:")
    print(f"{'Slack Time (s)':<20} {'Success Rate':<15} {'Treated/Total'}")
    print("-" * 55)
    
    for slack in slack_times:
        config.slack_time = slack
        planner = PathPlanner(config, PlanningAlgorithm.GREEDY)
        
        # 分配杂草到左右工作区
        left_weeds = []
        right_weeds = []
        for weed in weeds:
            center_x, center_y = weed.get_center()
            if config.left_galvo.is_in_work_zone(center_x, center_y):
                left_weeds.append(weed)
            elif config.right_galvo.is_in_work_zone(center_x, center_y):
                right_weeds.append(weed)
        
        # 规划路径
        left_plan = planner.plan(left_weeds, 0.0, config.left_galvo) if left_weeds else []
        right_plan = planner.plan(right_weeds, 0.0, config.right_galvo) if right_weeds else []
        
        total = len(left_weeds) + len(right_weeds)
        treated = len(left_plan) + len(right_plan)
        success_rate = treated / total if total > 0 else 0
        
        print(f"{slack:<20.2f} {success_rate:<15.1%} {treated}/{total}")


def example_5_custom_config():
    """示例5：自定义系统配置"""
    print("\n" + "="*60)
    print("Example 5: Custom System Configuration")
    print("="*60)
    
    # 创建默认配置
    config = create_default_config()
    
    # 修改参数
    config.platform_velocity = 100.0  # 增加平台速度到100 mm/s
    config.slack_time = 0.1  # 增加slack时间到0.1s
    config.laser_treatment_time = 0.05  # 减少处理时间到0.05s
    
    # 增加振镜速度
    config.left_galvo.max_angular_velocity = 10.0
    config.right_galvo.max_angular_velocity = 10.0
    
    print("Custom Configuration:")
    print(f"  Platform Velocity: {config.platform_velocity} mm/s")
    print(f"  Slack Time: {config.slack_time} s")
    print(f"  Treatment Time: {config.laser_treatment_time} s")
    print(f"  Max Angular Velocity: {config.left_galvo.max_angular_velocity} rad/s")
    
    # 测试新配置
    np.random.seed(42)
    weeds = generate_random_weeds(30)
    
    results = compare_algorithms(config, weeds, 'example_5_custom_config.png')
    
    print("\nResults with custom config:")
    for algo, result in results.items():
        print(f"  {algo}: {result['success_rate']:.1%} ({result['treated']}/{result['total']})")


def main():
    """运行所有示例"""
    print("\n" + "="*60)
    print("Laser Weeding Path Planning Examples")
    print("激光除草路径规划示例")
    print("="*60)
    
    # 运行所有示例
    example_1_basic_usage()
    example_2_compare_algorithms()
    example_3_custom_weeds()
    example_4_test_slack_times()
    example_5_custom_config()
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("所有示例运行完成！")
    print("="*60)


if __name__ == "__main__":
    main()
