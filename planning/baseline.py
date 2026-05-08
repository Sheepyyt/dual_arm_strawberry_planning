
#!/usr/bin/env python3
"""
baseline.py

Baseline 模式示例。
兼容两种导入方式：
- 新版 dual_arm_planner.py 中若保留了 DualArmPlanner 别名，则直接使用
- 若没有该别名，则回退到 BaselinePlanner
"""

import os
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

try:
    from dual_arm_planner import DualArmPlanner
except ImportError:
    from dual_arm_planner import BaselinePlanner as DualArmPlanner


def main():
    print("Dual-Arm Strawberry Harvesting — Baseline Mode")
    print("=" * 60)

    np.random.seed(0)

    planner = DualArmPlanner()
    planner.create_task_dataset({
        "B1": 5,
        "B2": 2,
        "B3": 2,
        "B4": 3,
        "B5": 2,
        "B6": 4,
    })

    heuristic_actions, milp_actions, improvement = planner.solve_optimization(
        time_limit=30,
        heuristic_name="spatial_order",
    )

    heuristic_makespan = max(a["end"] for a in heuristic_actions) if heuristic_actions else 0.0
    milp_makespan = max(a["end"] for a in milp_actions) if milp_actions else heuristic_makespan

    print(f"\nHeuristic makespan: {heuristic_makespan:.2f}s")
    print(f"MILP makespan:      {milp_makespan:.2f}s")
    print(f"Improvement:        {improvement:.1f}%")

    result_dir = os.path.join(SCRIPT_DIR, "results", "baseline_run")
    planner.save_results(result_dir)

    print("\nDone. Check planning/results/baseline_run/ for outputs.")


if __name__ == "__main__":
    main()
