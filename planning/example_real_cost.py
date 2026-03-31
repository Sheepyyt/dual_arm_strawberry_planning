#!/usr/bin/env python3
"""
example_real_cost.py — Real Cost 模式使用示例

演示如何使用 RealCostPlanner 进行双臂草莓采摘规划：
  * 机械臂定义来源：dual_arm_ik_xy_centered.urdf（与 build_roi_table.py 完全一致）
  * 单程移动代价来源：roi/results/dual_arm_cost.pkl 中的 cost table
  * 草莓采样方式：仅从 cost table 可达网格离散点中筛选，且位于 B1-B6 区域内
"""

import os
import numpy as np
from dual_arm_planner import RealCostPlanner

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def example_real_cost_random():
    """示例 1：从 cost table 可达网格随机采样草莓任务，然后求解。"""
    print("=== Real Cost Example 1: Grid-Sampled Task Generation ===")
    np.random.seed(42)

    # 创建 RealCostPlanner 实例（自动加载 URDF 和 cost table 默认路径）
    planner = RealCostPlanner(base_operation_time=0.0)

    # 打印 URDF 读取到的机械臂基座位置
    print(f"Left arm base  (from URDF): {planner.L_base}")
    print(f"Right arm base (from URDF): {planner.R_base}")

    # 定义每个区域的草莓数量
    points_per_region = {
        "B1": 4,   # R 独占区（vehicle frame 负 x 侧，R 臂基座在负 x）
        "B2": 3,   # 干涉区（正 y 侧）
        "B3": 4,   # L 独占区（vehicle frame 正 x 侧，L 臂基座在正 x）
        "B4": 3,   # R 独占区（vehicle frame 负 x 侧）
        "B5": 3,   # 干涉区（负 y 侧）
        "B6": 4,   # L 独占区（vehicle frame 正 x 侧）
    }

    # 从 cost table 可达网格点采样
    task_df = planner.create_task_dataset(points_per_region)
    print(f"\n已生成 {len(task_df)} 个草莓任务")
    print(task_df[['region', 'x', 'y', 'accessible_by', 'time_to_L', 'time_to_R']].to_string())

    # 运行优化（启发式 + MILP）
    print("\n=== Running optimization ===")
    heuristic_actions, milp_actions, improvement = planner.solve_optimization(
        time_limit=30,
        heuristic_name="spatial_order",
    )

    heuristic_makespan = max(a["end"] for a in heuristic_actions)
    milp_makespan      = max(a["end"] for a in milp_actions)
    print(f"\nHeuristic makespan : {heuristic_makespan:.3f}s")
    print(f"MILP makespan      : {milp_makespan:.3f}s")
    print(f"Improvement        : {improvement:.1f}%")

    # 保存结果
    result_dir = os.path.join(SCRIPT_DIR, "results", "real_cost_example1")
    planner.save_results(result_dir)
    print(f"\n结果已保存到 {result_dir}/")

    return planner


def example_real_cost_custom_paths():
    """示例 2：显式指定 URDF 和 cost table 路径。"""
    print("\n=== Real Cost Example 2: Explicit Paths ===")
    np.random.seed(0)

    repo_root = os.path.dirname(SCRIPT_DIR)
    cost_table_path = os.path.join(repo_root, "roi", "results", "dual_arm_cost.pkl")
    urdf_path       = os.path.join(repo_root, "urdf", "dual_arm_ik_xy_centered.urdf")

    planner = RealCostPlanner(
        cost_table_path=cost_table_path,
        urdf_path=urdf_path,
        base_operation_time=0.5,  # 每颗草莓额外 0.5s 固定操作时间
    )

    points_per_region = {"B1": 3, "B2": 2, "B3": 3, "B4": 2, "B5": 2, "B6": 3}
    task_df = planner.create_task_dataset(points_per_region)
    print(f"已生成 {len(task_df)} 个草莓任务（含 0.5s 固定处理时间）")

    heuristic_actions, milp_actions, improvement = planner.solve_optimization(time_limit=20)

    heuristic_makespan = max(a["end"] for a in heuristic_actions)
    milp_makespan      = max(a["end"] for a in milp_actions)
    print(f"Heuristic makespan : {heuristic_makespan:.3f}s")
    print(f"MILP makespan      : {milp_makespan:.3f}s")
    print(f"Improvement        : {improvement:.1f}%")

    result_dir = os.path.join(SCRIPT_DIR, "results", "real_cost_example2")
    planner.save_results(result_dir)
    print(f"结果已保存到 {result_dir}/")

    return planner


if __name__ == "__main__":
    print("Dual-Arm Real Cost Planner Examples")
    print("=" * 50)

    planner1 = example_real_cost_random()

    # 示例 2 可选运行（默认注释掉）
    # planner2 = example_real_cost_custom_paths()

    print("\n" + "=" * 50)
    print("All examples completed!")
    print("Generated files:")
    print("  results/real_cost_example1/heuristic_animation.gif")
    print("  results/real_cost_example1/milp_animation.gif")
    print("  results/real_cost_example1/heuristic_gantt.png")
    print("  results/real_cost_example1/milp_gantt.png")
    print("  results/real_cost_example1/comparison_summary.txt")
