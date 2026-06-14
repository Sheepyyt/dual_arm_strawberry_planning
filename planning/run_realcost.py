#!/usr/bin/env python3
"""
run_realcost.py — point-motion Real Cost 模式入口（当前主流程）
=============================================================
读取 ompl/results/point_table/point_table.pkl 作为优化器输入，
跑启发式 + MILP，输出甘特图 / 任务地图 / 动画 / summary 到 planning/results/。

运行环境：PickPlan（需要 gurobipy + 有效 Gurobi 许可证）
运行方式：
    conda activate PickPlan
    python planning/run_realcost.py
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from realcost_planner import RealCostPlanner


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


def example_real_cost_specified():
    print("\n=== Point-motion Real Cost Example 2: Specified task locations ===\n")
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


if __name__ == "__main__":
    print("Dual-Arm Strawberry Harvesting — Point-motion Real Cost")
    print("=" * 70)

    example_real_cost_all_points()
    # example_real_cost_specified()

    print("\n" + "=" * 70)
    print("Done. Check planning/results/ for outputs.")
