#!/usr/bin/env python3
"""
real_cost_example.py — 'Real Cost' 模式使用示例
=====================================================
演示如何使用 RealCostPlanner（基于 URDF 机械臂定义 + cost table 真实代价）
对固定停车点下的双臂草莓采摘任务进行规划与优化。

运行前请确保以下文件存在：
  - urdf/dual_arm_ik_xy_centered.urdf
  - roi/results/dual_arm_cost.pkl

若 cost table 不存在，请先运行：
  cd roi && python3 build_roi_table.py
"""

import os
import sys
import numpy as np

# 将 planning/ 目录加入路径（支持从项目根目录运行）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from dual_arm_planner import RealCostPlanner


# ============================================================
# 示例 1：随机从 cost table 网格采样任务
# ============================================================
def example_real_cost_random():
    print("=== Real Cost Example 1: Grid-Sampled Random Tasks ===\n")
    np.random.seed(0)

    # 初始化 RealCostPlanner（URDF 与 cost table 路径使用默认值）
    planner = RealCostPlanner()

    # 在每个区域内从 cost table 可达网格点中随机采样
    points_per_region = {
        "B1": 5,   # 仅左臂区域
        "B2": 3,   # 干涉区（按点实际可达性分配）
        "B3": 5,   # 仅右臂区域
        "B4": 4,   # 仅左臂区域
        "B5": 3,   # 干涉区
        "B6": 4,   # 仅右臂区域
    }

    task_df = planner.create_task_dataset(points_per_region)
    print(f"Created {len(task_df)} tasks from cost table grid points.")
    print(task_df.to_string())
    print()

    # 运行启发式 + MILP 联合优化
    print("Running optimization...")
    heuristic_actions, milp_actions, improvement = planner.solve_optimization(
        time_limit=30,
        heuristic_name="spatial_order",
    )

    heuristic_makespan = max(a["end"] for a in heuristic_actions)
    milp_makespan = max(a["end"] for a in milp_actions) if milp_actions else heuristic_makespan

    print(f"\nHeuristic makespan: {heuristic_makespan:.2f}s")
    print(f"MILP makespan:      {milp_makespan:.2f}s")
    print(f"Improvement:        {improvement:.1f}%")

    # 保存结果到 planning/results/real_cost_example1/
    result_dir = os.path.join(SCRIPT_DIR, "results", "real_cost_example1")
    planner.save_results(result_dir)

    return planner


# ============================================================
# 示例 2：指定 cost table 网格点位置
# ============================================================
def example_real_cost_specified():
    print("\n=== Real Cost Example 2: Specified Grid-Point Tasks ===\n")

    planner = RealCostPlanner()

    # 手动指定任务位置（必须是 cost table 网格点：step=0.02m，z=0.560）
    # B1/B4：L-only；B3/B6：R-only；B2/B5：按 cost table 实际可达性
    task_locations = [
        # B1 任务（仅左臂）
        {"x": -0.40, "y": 0.31, "region": "B1"},
        {"x": -0.30, "y": 0.49, "region": "B1"},
        {"x": -0.20, "y": 0.55, "region": "B1"},

        # B2 任务（干涉区，按点可达性决定）
        {"x":  0.06, "y": 0.43, "region": "B2"},
        {"x": -0.10, "y": 0.59, "region": "B2"},
        {"x":  0.12, "y": 0.57, "region": "B2"},

        # B3 任务（仅右臂）
        {"x":  0.24, "y": 0.57, "region": "B3"},
        {"x":  0.16, "y": 0.53, "region": "B3"},

        # B4 任务（仅左臂）
        {"x": -0.24, "y": -0.57, "region": "B4"},
        {"x": -0.18, "y": -0.53, "region": "B4"},

        # B5 任务（干涉区）
        {"x":  0.08, "y": -0.39, "region": "B5"},
        {"x": -0.12, "y": -0.57, "region": "B5"},

        # B6 任务（仅右臂）
        {"x":  0.16, "y": -0.53, "region": "B6"},
        {"x":  0.18, "y": -0.57, "region": "B6"},
    ]

    task_df = planner.load_task_locations(task_locations)
    print(f"Loaded {len(task_df)} tasks.")
    print(task_df.to_string())
    print()

    print("Running optimization...")
    heuristic_actions, milp_actions, improvement = planner.solve_optimization(
        time_limit=30,
        heuristic_name="spatial_order",
    )

    heuristic_makespan = max(a["end"] for a in heuristic_actions)
    milp_makespan = max(a["end"] for a in milp_actions) if milp_actions else heuristic_makespan

    print(f"\nHeuristic makespan: {heuristic_makespan:.2f}s")
    print(f"MILP makespan:      {milp_makespan:.2f}s")
    print(f"Improvement:        {improvement:.1f}%")

    result_dir = os.path.join(SCRIPT_DIR, "results", "real_cost_example2")
    planner.save_results(result_dir)

    return planner


# ============================================================
# 示例 3：对比 Baseline 与 Real Cost 的调度结果
# ============================================================
def example_compare_modes():
    print("\n=== Real Cost Example 3: Compare Baseline vs Real Cost ===\n")
    np.random.seed(0)

    from dual_arm_planner import BaselinePlanner

    # ---- Baseline 模式 ----
    baseline = BaselinePlanner()
    baseline.create_task_dataset({"B1": 4, "B2": 3, "B3": 4, "B4": 4, "B5": 3, "B6": 4})
    baseline_h, baseline_m, _ = baseline.solve_optimization(time_limit=20)
    baseline_makespan = max(a["end"] for a in baseline_m) if baseline_m else max(a["end"] for a in baseline_h)

    # ---- Real Cost 模式 ----
    np.random.seed(0)
    real_cost = RealCostPlanner()
    real_cost.create_task_dataset({"B1": 4, "B2": 3, "B3": 4, "B4": 4, "B5": 3, "B6": 4})
    rc_h, rc_m, _ = real_cost.solve_optimization(time_limit=20)
    rc_makespan = max(a["end"] for a in rc_m) if rc_m else max(a["end"] for a in rc_h)

    print("\n=== Mode Comparison ===")
    print(f"{'Mode':<15} | {'Makespan (s)':<14} | {'Cost Model':<25}")
    print("-" * 60)
    print(f"{'Baseline':<15} | {baseline_makespan:<14.2f} | Euclidean distance")
    print(f"{'Real Cost':<15} | {rc_makespan:<14.2f} | IK joint-space cost table")


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    print("Dual-Arm Strawberry Harvesting — Real Cost Mode Examples")
    print("=" * 60)

    # 示例 1：随机网格采样任务
    planner1 = example_real_cost_random()

    # 示例 2：指定网格点任务
    # planner2 = example_real_cost_specified()

    # 示例 3：与 Baseline 对比（运行时间较长）
    # example_compare_modes()

    print("\n" + "=" * 60)
    print("All examples completed! Check planning/results/ for outputs.")
    print("Generated files:")
    print("  - *_animation.gif  : Task execution animations")
    print("  - *_gantt.png      : Gantt chart visualizations")
    print("  - comparison_summary.txt : Detailed comparison results")
