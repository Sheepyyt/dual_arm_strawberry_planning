#!/usr/bin/env python3
"""
real_cost.py — Real-cost mode example script.

Uses dual_arm_cost.pkl to supply real IK-based one-way costs and true
accessibility, instead of the Euclidean-distance approximation in
baseline.py.

区域划分规则（vehicle frame）：
  - left_only          : 仅左臂可达（非干涉区）
  - right_only         : 仅右臂可达（非干涉区）
  - interference_left  : 双臂均可达，y ≥ 0（车辆左侧，独立干涉区）
  - interference_right : 双臂均可达，y < 0 （车辆右侧，独立干涉区）

两个干涉区物理上由 y=0 车辆中心线隔开，各自独立串行（互不影响效率）。

Usage:
    python real_cost.py

Results saved to:
    planning/results/real_cost_basic/
    planning/results/real_cost_per_region/   (按区域独立指定采样数)
"""
import os
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)

# Resolve default file paths relative to the repository root
DEFAULT_COST_PKL = os.path.join(REPO_ROOT, "roi",  "results", "dual_arm_cost.pkl")
DEFAULT_URDF     = os.path.join(REPO_ROOT, "urdf", "dual_arm_ik.urdf")

# Make sure dual_arm_planner is importable when running this file directly
sys.path.insert(0, SCRIPT_DIR)
from dual_arm_planner import RealCostPlanner


# --------------------------------------------------------------------------- #
# 示例 1：基本用法（每个区域各 6 个采样点）
# --------------------------------------------------------------------------- #

def example_real_cost_basic(cost_pkl: str = DEFAULT_COST_PKL,
                             urdf_path: str = DEFAULT_URDF) -> RealCostPlanner:
    """
    基本示例：从 4 个区域各采样 6 个任务点，运行启发式 + MILP 优化，保存结果。
    """
    print("=== Real-Cost Mode: Basic Example ===\n")

    # 1. 初始化规划器
    #    基座位置从 URDF vehicle_to_left_arm / vehicle_to_right_arm 自动解析
    planner = RealCostPlanner(
        urdf_path=urdf_path,
        cost_pkl_path=cost_pkl,
    )

    print(f"Left arm base  (vehicle frame x,y): {planner.L_base}")
    print(f"Right arm base (vehicle frame x,y): {planner.R_base}\n")

    # 2. 从 cost table 按区域采样
    #    n_per_region=6 → 每个区域各采 6 个点（4 个区域共最多 24 个任务）
    task_df = planner.sample_tasks_from_cost_table(n_per_region=6, seed=42)

    print(f"\nTask DataFrame ({len(task_df)} rows, first 8 shown):")
    print(task_df[["region", "x", "y", "z",
                   "accessible_by", "is_interference",
                   "interference_group",
                   "time_to_L", "time_to_R"]].head(8).to_string())

    # 3. 运行优化（启发式热启动 + MILP 精化）
    print("\n=== Running optimization ===")
    heuristic_actions, milp_actions, improvement = planner.solve_optimization(
        time_limit=30,
        heuristic_name="spatial_order",
    )

    # 4. 保存结果（动画 GIF + 甘特图 PNG + 文本摘要）
    result_dir = os.path.join(SCRIPT_DIR, "results", "real_cost_basic")
    planner.save_results(result_dir)

    return planner


# --------------------------------------------------------------------------- #
# 示例 2：按区域独立指定采样数
# --------------------------------------------------------------------------- #

def example_real_cost_per_region(cost_pkl: str = DEFAULT_COST_PKL,
                                  urdf_path: str = DEFAULT_URDF) -> RealCostPlanner:
    """
    按区域独立指定采样数示例：两个干涉区各采 8 个，单臂区各采 4 个。
    """
    print("\n=== Real-Cost Mode: Per-Region Sampling ===\n")

    planner = RealCostPlanner(
        urdf_path=urdf_path,
        cost_pkl_path=cost_pkl,
    )

    task_df = planner.sample_tasks_from_cost_table(
        n_per_region={
            "left_only":          4,
            "right_only":         4,
            "interference_left":  8,
            "interference_right": 8,
        },
        seed=0,
    )

    print(f"\nSampled {len(task_df)} tasks.")
    print(task_df[["region", "accessible_by", "is_interference",
                   "interference_group",
                   "time_to_L", "time_to_R"]].head(8).to_string())

    heuristic_actions, milp_actions, improvement = planner.solve_optimization(
        time_limit=30,
        heuristic_name="spatial_order",
    )

    result_dir = os.path.join(SCRIPT_DIR, "results", "real_cost_per_region")
    planner.save_results(result_dir)

    return planner


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    print("Dual-Arm Real-Cost Planner")
    print("=" * 50)

    # 示例 1：基本用法（每区域 6 个采样点）
    planner1 = example_real_cost_basic()

    # 示例 2：按区域独立指定采样数（可选，取消注释以运行）
    # planner2 = example_real_cost_per_region()

    print("\n" + "=" * 50)
    print("Done. Results saved under planning/results/real_cost_*/")
    print("Files generated:")
    print("  - *_animation.gif  : task execution animations")
    print("  - *_gantt.png      : Gantt chart visualizations")
    print("  - comparison_summary.txt : solver comparison")
