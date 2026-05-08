#!/usr/bin/env python3
"""
real_cost_example.py — point-motion Real Cost 模式示例
=====================================================
说明：
- Baseline 模式保持不变，仍在 dual_arm_planner.py 中
- 当前 RealCostPlanner 已替换为 point-motion 版本
- 它读取：
    ompl/results/point_motion_table/point_motion_table_selected_points.pkl
  作为优化器输入
"""

import os
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from dual_arm_planner import RealCostPlanner, BaselinePlanner


def example_real_cost_all_points():
    print("=== Point-motion Real Cost Example 1: All non-discard points ===\n")
    planner = RealCostPlanner()

    task_df = planner.create_task_dataset(points_per_label=None, random_seed=0)
    print(f"Loaded {len(task_df)} optimizer-ready tasks.")
    print(task_df[[
        "key", "x", "y", "label", "allowed_arms", "must_assign_to",
        "time_to_L", "time_to_R", "candidate_mode_L", "candidate_mode_R"
    ]].to_string())
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

    result_dir = os.path.join(SCRIPT_DIR, "results", "real_cost_example_all_points")
    planner.save_results(result_dir)
    return planner


def example_real_cost_sampled():
    print("\n=== Point-motion Real Cost Example 2: Sampled by label ===\n")
    planner = RealCostPlanner()

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
        "time_to_L", "time_to_R"
    ]].to_string())
    print()

    print("Running optimization...")
    heuristic_actions, milp_actions, improvement = planner.solve_optimization(
        time_limit=20,
        heuristic_name="spatial_order",
    )

    heuristic_makespan = max(a["end"] for a in heuristic_actions)
    milp_makespan = max(a["end"] for a in milp_actions) if milp_actions else heuristic_makespan

    print(f"\nHeuristic makespan: {heuristic_makespan:.2f}s")
    print(f"MILP makespan:      {milp_makespan:.2f}s")
    print(f"Improvement:        {improvement:.1f}%")

    result_dir = os.path.join(SCRIPT_DIR, "results", "real_cost_example_sampled")
    planner.save_results(result_dir)
    return planner


def example_real_cost_specified():
    print("\n=== Point-motion Real Cost Example 3: Specified task locations ===\n")
    planner = RealCostPlanner()

    # 直接指定点（无需再给 B1-B6 region）
    task_locations = [
        {"x": -0.1, "y": 0.5, "z": 0.56},
        {"x": 0.0, "y": 0.4, "z": 0.56},
        {"x": 0.1, "y": -0.5, "z": 0.56},
        {"x": 0.2, "y": 0.3, "z": 0.56},
    ]
    task_df = planner.load_task_locations(task_locations)
    print(f"Loaded {len(task_df)} specified tasks.")
    print(task_df[[
        "key", "x", "y", "label", "allowed_arms", "must_assign_to",
        "time_to_L", "time_to_R"
    ]].to_string())
    print()

    print("Running optimization...")
    heuristic_actions, milp_actions, improvement = planner.solve_optimization(
        time_limit=20,
        heuristic_name="spatial_order",
    )

    heuristic_makespan = max(a["end"] for a in heuristic_actions)
    milp_makespan = max(a["end"] for a in milp_actions) if milp_actions else heuristic_makespan

    print(f"\nHeuristic makespan: {heuristic_makespan:.2f}s")
    print(f"MILP makespan:      {milp_makespan:.2f}s")
    print(f"Improvement:        {improvement:.1f}%")

    result_dir = os.path.join(SCRIPT_DIR, "results", "real_cost_example_specified")
    planner.save_results(result_dir)
    return planner


def example_compare_baseline_and_real_cost():
    print("\n=== Point-motion Real Cost Example 4: Compare Baseline vs Real Cost ===\n")
    np.random.seed(0)

    baseline = BaselinePlanner()
    baseline.create_task_dataset({"B1": 4, "B2": 3, "B3": 4, "B4": 4, "B5": 3, "B6": 4})
    b_h, b_m, _ = baseline.solve_optimization(time_limit=20)
    b_ms = max(a["end"] for a in b_m) if b_m else max(a["end"] for a in b_h)

    real_cost = RealCostPlanner()
    real_cost.create_task_dataset(points_per_label={"parallel": 12, "serial_upper": 4, "serial_lower": 4}, random_seed=0)
    rc_h, rc_m, _ = real_cost.solve_optimization(time_limit=20)
    rc_ms = max(a["end"] for a in rc_m) if rc_m else max(a["end"] for a in rc_h)

    print("\n=== Mode Comparison ===")
    print(f"{'Mode':<18} | {'Makespan (s)':<14} | {'Cost Model':<35}")
    print("-" * 80)
    print(f"{'Baseline':<18} | {b_ms:<14.2f} | Euclidean distance + B1-B6")
    print(f"{'Point-motion RealCost':<18} | {rc_ms:<14.2f} | OMPL-derived point-motion table")


if __name__ == "__main__":
    print("Dual-Arm Strawberry Harvesting — Point-motion Real Cost Examples")
    print("=" * 70)

    example_real_cost_all_points()
    # example_real_cost_sampled()
    # example_real_cost_specified()
    # example_compare_baseline_and_real_cost()

    print("\n" + "=" * 70)
    print("Done. Check planning/results/ for outputs.")
