#!/usr/bin/env python3
"""
real_cost.py — Real-cost mode example script.

Uses dual_arm_cost.pkl to supply real IK-based one-way costs and true
accessibility, instead of the Euclidean-distance approximation in
baseline.py.

Usage:
    python real_cost.py

The script instantiates RealCostPlanner, samples tasks directly from
the cost-table grid points, runs the heuristic + MILP solver, and saves
results (animations, Gantt charts, summary) to:
    planning/results/real_cost_basic/
    planning/results/real_cost_custom_windows/    (commented-out by default)
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
# 示例 1：基本用法（4 默认窗口，每窗口 6 个采样点）
# --------------------------------------------------------------------------- #

def example_real_cost_basic(cost_pkl: str = DEFAULT_COST_PKL,
                             urdf_path: str = DEFAULT_URDF) -> RealCostPlanner:
    """
    基本示例：从 cost table 采样 24 个任务点，运行启发式 + MILP 优化，保存结果。
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

    # 2. 从 cost table 网格点直接采样任务
    #    n_per_window=6 → 每个窗口 6 个点，4 个窗口共最多 24 个任务
    task_df = planner.sample_tasks_from_cost_table(n_per_window=6, seed=42)

    print(f"\nTask DataFrame ({len(task_df)} rows, first 8 shown):")
    print(task_df[["region", "x", "y", "z",
                   "accessible_by", "is_interference",
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
# 示例 2：自定义窗口（2 个大窗口，每窗口 8 个采样点）
# --------------------------------------------------------------------------- #

def example_real_cost_custom_windows(cost_pkl: str = DEFAULT_COST_PKL,
                                     urdf_path: str = DEFAULT_URDF) -> RealCostPlanner:
    """
    自定义窗口示例：合并左/右侧为各 1 个大窗口，每窗口采样 8 个点。

    展示如何通过 windows_config 覆盖默认 4 窗口配置，
    以及如何通过字典为不同窗口指定不同的采样数量。
    """
    print("\n=== Real-Cost Mode: Custom Windows ===\n")

    custom_windows = [
        {"name": "W_left",  "x_min": -0.65, "x_max": 0.30, "y_min":  0.30, "y_max":  0.60},
        {"name": "W_right", "x_min": -0.65, "x_max": 0.30, "y_min": -0.60, "y_max": -0.30},
    ]

    planner = RealCostPlanner(
        urdf_path=urdf_path,
        cost_pkl_path=cost_pkl,
        windows_config=custom_windows,
    )

    task_df = planner.sample_tasks_from_cost_table(
        n_per_window={"W_left": 8, "W_right": 8},
        seed=0,
    )

    print(f"\nSampled {len(task_df)} tasks.")
    print(task_df[["region", "accessible_by", "is_interference",
                   "time_to_L", "time_to_R"]].head(8).to_string())

    heuristic_actions, milp_actions, improvement = planner.solve_optimization(
        time_limit=30,
        heuristic_name="spatial_order",
    )

    result_dir = os.path.join(SCRIPT_DIR, "results", "real_cost_custom_windows")
    planner.save_results(result_dir)

    return planner


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    print("Dual-Arm Real-Cost Planner")
    print("=" * 50)

    # 示例 1：基本用法（默认 4 窗口）
    planner1 = example_real_cost_basic()

    # 示例 2：自定义窗口（可选，取消注释以运行）
    # planner2 = example_real_cost_custom_windows()

    print("\n" + "=" * 50)
    print("Done. Results saved under planning/results/real_cost_*/")
    print("Files generated:")
    print("  - *_animation.gif  : task execution animations")
    print("  - *_gantt.png      : Gantt chart visualizations")
    print("  - comparison_summary.txt : solver comparison")
