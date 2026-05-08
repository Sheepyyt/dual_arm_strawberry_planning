
#!/usr/bin/env python3
"""
real_cost_point_motion_example.py

示例：使用 point-motion 版 RealCostPlanner
=====================================================
- Baseline 模式不变，仍在 dual_arm_planner.py 中
- 这个脚本演示新的 point-motion RealCost 模式如何直接读取
  ompl/results/point_motion_table/point_motion_table_selected_points.pkl
  进行调度优化

建议放在 planning/ 目录下运行。
"""

import os
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from dual_arm_planner_point_motion import RealCostPlanner


def example_all_points():
    print("=== Point-motion RealCost Example: all non-discard points ===")
    planner = RealCostPlanner()

    # 直接加载全部非 discard 点
    task_df = planner.create_task_dataset(points_per_label=None, random_seed=0)
    print(f"Loaded {len(task_df)} optimizer-ready tasks.")
    print(task_df[[
        "key", "x", "y", "label", "allowed_arms", "must_assign_to",
        "one_way_cost_L", "one_way_cost_R", "time_to_L", "time_to_R"
    ]].to_string())

    heuristic_actions, milp_actions, improvement = planner.solve_optimization(
        time_limit=30,
        heuristic_name="spatial_order",
    )

    heuristic_makespan = max(a["end"] for a in heuristic_actions) if heuristic_actions else 0.0
    milp_makespan = max(a["end"] for a in milp_actions) if milp_actions else heuristic_makespan

    print(f"\nHeuristic makespan: {heuristic_makespan:.2f}s")
    print(f"MILP makespan:      {milp_makespan:.2f}s")
    print(f"Improvement:        {improvement:.1f}%")

    result_dir = os.path.join(SCRIPT_DIR, "results", "real_cost_point_motion_all")
    planner.save_results(result_dir)
    return planner


def example_sampled_points():
    print("=== Point-motion RealCost Example: sampled by label ===")
    planner = RealCostPlanner()

    # 按标签抽样，便于先快速测试
    task_df = planner.create_task_dataset(
        points_per_label={
            "parallel": 12,
            "serial_upper": 4,
            "serial_lower": 4,
        },
        random_seed=42,
    )
    print(f"Loaded {len(task_df)} sampled tasks.")
    print(task_df[[
        "key", "x", "y", "label", "allowed_arms", "must_assign_to",
        "one_way_cost_L", "one_way_cost_R", "time_to_L", "time_to_R"
    ]].to_string())

    heuristic_actions, milp_actions, improvement = planner.solve_optimization(
        time_limit=20,
        heuristic_name="spatial_order",
    )

    heuristic_makespan = max(a["end"] for a in heuristic_actions) if heuristic_actions else 0.0
    milp_makespan = max(a["end"] for a in milp_actions) if milp_actions else heuristic_makespan

    print(f"\nHeuristic makespan: {heuristic_makespan:.2f}s")
    print(f"MILP makespan:      {milp_makespan:.2f}s")
    print(f"Improvement:        {improvement:.1f}%")

    result_dir = os.path.join(SCRIPT_DIR, "results", "real_cost_point_motion_sampled")
    planner.save_results(result_dir)
    return planner


if __name__ == "__main__":
    print("Dual-Arm Strawberry Harvesting — Point-motion RealCost Mode")
    print("=" * 70)

    example_all_points()
    # example_sampled_points()

    print("\nDone. Check planning/results/real_cost_point_motion_* for outputs.")
