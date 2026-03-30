#!/usr/bin/env python3
from dual_arm_planner import DualArmPlanner, BaselinePlanner, RealCostPlanner
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
import numpy as np


# 示例 1：随机生成任务位置
def example_with_random_tasks():
    print("=== Example 1: Random Task Generation ===")
    np.random.seed(0)   # 设置随机种子以确保可重复性
    
    # 创建优化器实例，它实际上调用了 dual_arm_harvesting_optimizer.py 文件里的 __init__ 函数（初始化函数）
    optimizer = DualArmPlanner()
    
    # 定义每个区域的任务数量
    points_per_region = {
        "B1": 5,   # 仅右臂区域（IK 统计：右臂可达 ~99%，左臂仅 28%）
        "B2": 2,   # 干涉区域（左 & 右）
        "B3": 2,   # 仅左臂区域（IK 统计：左臂可达 ~77%，右臂 0%）
        "B4": 3,   # 仅右臂区域
        "B5": 2,   # 干涉区域（左 & 右）
        "B6": 4    # 仅左臂区域
    }
    
    # 在每个区域的矩形范围内，随机生成对应数量的草莓坐标和姿态，并算出这些草莓分别对于左右臂的处理时间
    task_df = optimizer.create_task_dataset(points_per_region)
    print(f"Created {len(task_df)} tasks")   # 打印创建的草莓任务总数
    print(task_df.head())   # 打印前几行任务数据以供参考
    """
    以下是输出示例（B1 现为仅右臂区域，显示 [R]）：
        创建了 18 颗草莓任务
          region         x         y accessible_by  time_to_L  time_to_R
        0     B1 -0.522003  0.494414           [R]        NaN   0.541969
        1     B1 -0.349238  0.635969           [R]        NaN   0.637872
        2     B1 -0.250345  0.450685           [R]        NaN   0.453413
    """
    
    # 使用优化方法求解：提供任务表，分别使用两种方法求解，输出方案和提升百分比
    print("\n=== Running optimization ===")
    heuristic_actions, milp_actions, improvement = optimizer.solve_optimization(
        time_limit=20,
        heuristic_name="spatial_order"
    )
    
    # 保存结果
    optimizer.save_results(os.path.join(SCRIPT_DIR, "results", "example1_results"))
    
    # 保存综合分析结果（包含详细图表及数据）
    # optimizer.save_comprehensive_results(os.path.join(SCRIPT_DIR, "results", "example1_comprehensive_results"))
    
    return optimizer


# 示例 2：自定义任务位置
def example_with_custom_tasks():
    print("\n=== Example 2: Custom Task Locations ===")
    
    # 创建带有自定义机械臂位置的优化器实例
    L_base = np.array([-0.25, 0])
    R_base = np.array([0.25, 0])
    optimizer = DualArmPlanner(L_base=L_base, R_base=R_base)
    
    # 自定义任务位置
    task_locations = [
        # B1 任务（仅右臂）
        {"x": -0.4, "y": 0.3, "region": "B1"},
        {"x": -0.5, "y": 0.4, "region": "B1"},
        {"x": -0.3, "y": 0.6, "region": "B1"},
        
        # B2 任务（干涉区域）
        {"x": 0.0, "y": 0.4, "region": "B2"},
        {"x": -0.1, "y": 0.5, "region": "B2"},
        {"x": 0.1, "y": 0.6, "region": "B2"},
        
        # B3 任务（仅左臂）
        {"x": 0.4, "y": 0.3, "region": "B3"},
        {"x": 0.5, "y": 0.4, "region": "B3"},
        
        # B4 任务（仅右臂）
        {"x": -0.4, "y": -0.3, "region": "B4"},
        {"x": -0.3, "y": -0.4, "region": "B4"},
        
        # B5 任务（干涉区域）
        {"x": 0.0, "y": -0.4, "region": "B5"},
        {"x": 0.1, "y": -0.5, "region": "B5"},
        
        # B6 任务（仅左臂）
        {"x": 0.4, "y": -0.3, "region": "B6"},
        {"x": 0.5, "y": -0.4, "region": "B6"},
    ]
    
    # 加载自定义任务
    task_df = optimizer.load_task_locations(task_locations)
    print(f"Loaded {len(task_df)} custom tasks")
    
    # 求解优化
    print("\n=== Running optimization for custom tasks ===")
    heuristic_actions, milp_actions, improvement = optimizer.solve_optimization(
        heuristic_name="spatial_order", 
        time_limit=20
    )

    # 保存结果
    optimizer.save_results(os.path.join(SCRIPT_DIR, "results", "example2_results"))
    
    return optimizer


# 示例 3：多场景分析
def example_analyze_multiple_scenarios():
    print("\n=== Example 3: Multiple Scenario Analysis ===")
    np.random.seed(0)   # 设置随机种子以确保可重复性
    
    # 定义三种不同的任务分布场景
    scenarios = [
        # 轻负载
        # {"name": "Light Load", "points": {"B1": 3, "B2": 5, "B3": 3, "B4": 3, "B5": 5, "B6": 3}},
        
        # 重干涉：干涉区域（B2, B5）草莓特别多
        {"name": "Heavy Interference", "points": {"B1": 5, "B2": 15, "B3": 5, "B4": 5, "B5": 15, "B6": 5}},
        
        # 负载不均：比如左边全是草莓，右边没几个
        # {"name": "Unbalanced", "points": {"B1": 10, "B2": 8, "B3": 2, "B4": 12, "B5": 6, "B6": 3}},
    ]
    results = []
    
    for scenario in scenarios:
        print(f"\n--- {scenario['name']} Scenario ---")
        
        # 初始化优化器
        optimizer = DualArmPlanner()
        task_df = optimizer.create_task_dataset(scenario['points'])
        
        heuristic_actions, milp_actions, improvement = optimizer.solve_optimization(
            time_limit=10  # 对于多个场景，时间限制较短
        )
        
        heuristic_makespan = max(action["end"] for action in heuristic_actions)
        milp_makespan = max(action["end"] for action in milp_actions) if milp_actions else heuristic_makespan
        
        results.append({
            "scenario": scenario['name'],
            "total_tasks": len(task_df),
            "heuristic_makespan": heuristic_makespan,
            "milp_makespan": milp_makespan,
            "improvement": improvement
        })
        
        # 保存每个场景的结果
        optimizer.save_comprehensive_results(
            result_dir="example3_comprehensive_results",
            experiment_name=scenario['name'].lower().replace(' ', '_')
        )
    
    # 打印比较结果
    print("\n=== Scenario Comparison ===")
    print("Scenario           | Tasks | Heuristic | MILP      | Improvement")
    print("-" * 65)
    for result in results:
        print(f"{result['scenario']:<18} | {result['total_tasks']:<5} | "
              f"{result['heuristic_makespan']:<9.2f} | {result['milp_makespan']:<9.2f} | "
              f"{result['improvement']:<6.1f}%")
    
    return results

# 示例 4：使用 RealCostPlanner（真实 cost table + 离散网格采样）
def example_with_real_cost():
    print("\n=== Example 4: Real Cost Planner (cost table lookup) ===")
    np.random.seed(42)

    # cost table 文件路径
    cost_table_path = os.path.join(SCRIPT_DIR, "..", "roi", "results", "dual_arm_cost.pkl")
    if not os.path.exists(cost_table_path):
        print(f"Cost table not found at {cost_table_path}. "
              "Please run roi/build_roi_table.py first to generate it.")
        return None

    # 创建 RealCostPlanner 实例（使用 URDF 的实际臂基座位置）
    optimizer = RealCostPlanner(cost_table_path=cost_table_path)

    # 每个区域采样任务数
    points_per_region = {
        "B1": 5,
        "B2": 2,
        "B3": 2,
        "B4": 3,
        "B5": 2,
        "B6": 2,
    }

    task_df = optimizer.create_task_dataset(points_per_region)
    print(f"Created {len(task_df)} tasks from cost table")
    print(task_df.head())

    # 统计各区域覆盖情况
    for region in ["B1", "B2", "B3", "B4", "B5", "B6"]:
        rdf = task_df[task_df["region"] == region]
        if not rdf.empty:
            left_count = rdf["accessible_by"].apply(lambda a: "L" in a).sum()
            right_count = rdf["accessible_by"].apply(lambda a: "R" in a).sum()
            print(f"  {region}: {len(rdf)} tasks  (L={left_count}, R={right_count})")

    print("\n=== Running optimization ===")
    heuristic_actions, milp_actions, improvement = optimizer.solve_optimization(
        time_limit=20,
        heuristic_name="spatial_order"
    )

    optimizer.save_results(os.path.join(SCRIPT_DIR, "results", "realcost_results"))
    return optimizer


if __name__ == "__main__":
    # 运行示例
    print("Dual-Arm Harvesting Optimizer Examples")
    print("=" * 50)

    # 示例 1：随机生成任务位置（Baseline，欧氏距离近似）
    optimizer1 = example_with_random_tasks()

    # 示例 2：自定义任务位置（Baseline）
    # optimizer2 = example_with_custom_tasks()

    # 示例 3：多场景分析（Baseline）
    # scenario_results = example_analyze_multiple_scenarios()

    # 示例 4：真实 cost table（RealCostPlanner）
    # optimizer4 = example_with_real_cost()

    print("\n" + "=" * 50)
    print("All examples completed! Check the result directories for outputs.")
    print("Files generated:")
    print("- *_animation.gif: Task execution animations")
    print("- *_gantt.png: Gantt chart visualizations")
    print("- comparison_summary.txt: Detailed comparison results")
