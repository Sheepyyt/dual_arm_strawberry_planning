#!/usr/bin/env python3
"""
real_cost_example.py
====================
RealCostPlanner 使用示例脚本。

RealCostPlanner 与 BaselinePlanner（DualArmPlanner）的主要差异：
  - 单程移动时间从预计算的 cost table（dual_arm_cost.pkl）查表获得，而非欧氏距离近似。
  - 任务点只从 cost table 已覆盖的离散网格点中筛选采样，不在连续空间随机落点。
  - 保守策略：B1/B4 强制 R-only，B3/B6 强制 L-only，B2/B5 允许实际可达臂。
  - 可视化时翻转 x 轴，使物理左臂（URDF x=+0.10866）在图中显示于左侧。

区域与臂对应关系（基于 IK 统计）：
  B1/B4 (负 x 侧)  ：右臂专属（右臂可达 ~91-99%，左臂仅约 28-30%）
  B2/B5 (中间区域) ：干涉区（双臂均有覆盖，但左臂覆盖率更高）
  B3/B6 (正 x 侧)  ：左臂专属（左臂可达 ~55-86%，右臂 0%）

用法：
  cd planning && python3 real_cost_example.py
"""

import os
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROI_DIR = os.path.join(SCRIPT_DIR, "..", "roi", "results")
COST_TABLE_PATH = os.path.join(ROI_DIR, "dual_arm_cost.pkl")

# 将 planning 目录加入路径，以便直接导入 dual_arm_planner
sys.path.insert(0, SCRIPT_DIR)
from dual_arm_planner import RealCostPlanner


# =============================================================================
# 工具：打印 task_df 的覆盖统计
# =============================================================================
def print_coverage_summary(task_df):
    print("\n  区域覆盖统计：")
    print(f"  {'Region':6} {'总数':6} {'L可达':6} {'R可达':6} {'is_interference':15}")
    for region in ["B1", "B2", "B3", "B4", "B5", "B6"]:
        rdf = task_df[task_df["region"] == region]
        if rdf.empty:
            continue
        l_count = rdf["accessible_by"].apply(lambda a: "L" in a).sum()
        r_count = rdf["accessible_by"].apply(lambda a: "R" in a).sum()
        interf = rdf["is_interference"].any()
        print(f"  {region:6} {len(rdf):6} {l_count:6} {r_count:6} {'✓' if interf else '':15}")


# =============================================================================
# 示例 1：从 cost table 随机采样任务并运行优化
# =============================================================================
def example_basic():
    print("=" * 60)
    print("RealCostPlanner 示例 1：基本用法（随机采样 + 优化）")
    print("=" * 60)

    if not os.path.exists(COST_TABLE_PATH):
        print(f"[错误] 未找到 cost table: {COST_TABLE_PATH}")
        print("请先运行 roi/build_roi_table.py 生成 dual_arm_cost.pkl。")
        return None

    np.random.seed(0)

    # 创建 RealCostPlanner（使用 URDF 实际臂基座位置作为默认值）
    planner = RealCostPlanner(cost_table_path=COST_TABLE_PATH)

    print(f"\n  臂基座位置（vehicle frame）：")
    print(f"    左臂 L_base = {planner.L_base}  (URDF: vehicle_to_left_arm)")
    print(f"    右臂 R_base = {planner.R_base}  (URDF: vehicle_to_right_arm)")
    print(f"    工作面 z   = {planner.z_work} m")

    # 从 cost table 离散网格点中采样任务
    points_per_region = {
        "B1": 4,   # 仅右臂区域
        "B2": 3,   # 干涉区
        "B3": 3,   # 仅左臂区域
        "B4": 4,   # 仅右臂区域
        "B5": 3,   # 干涉区
        "B6": 3,   # 仅左臂区域
    }

    task_df = planner.create_task_dataset(points_per_region)
    print(f"\n  成功从 cost table 采样 {len(task_df)} 个任务点")
    print("\n  task_df 前 6 行：")
    print(task_df[["region", "x", "y", "accessible_by", "is_interference",
                   "time_to_L", "time_to_R"]].head(6).to_string(index=False))
    print_coverage_summary(task_df)

    # 求解
    print("\n  运行优化...")
    h_actions, m_actions, improvement = planner.solve_optimization(time_limit=20)

    h_makespan = max(a["end"] for a in h_actions)
    m_makespan = max(a["end"] for a in m_actions)
    print(f"\n  启发式 makespan : {h_makespan:.3f} s")
    print(f"  MILP makespan   : {m_makespan:.3f} s")
    print(f"  提升             : {improvement:.1f}%")

    # 保存结果
    result_dir = os.path.join(SCRIPT_DIR, "results", "realcost_example1")
    planner.save_results(result_dir)
    print(f"\n  结果已保存到 {result_dir}/")

    return planner


# =============================================================================
# 示例 2：手动指定任务坐标（需与 cost table 网格对齐）
# =============================================================================
def example_custom_locations():
    print("\n" + "=" * 60)
    print("RealCostPlanner 示例 2：手动指定任务坐标")
    print("=" * 60)

    if not os.path.exists(COST_TABLE_PATH):
        print(f"[错误] 未找到 cost table: {COST_TABLE_PATH}")
        return None

    planner = RealCostPlanner(cost_table_path=COST_TABLE_PATH)

    # 提示：坐标会自动做 grid snap（步长 0.02 m），对齐 cost table 键格式。
    # 若所在网格点不在 cost table 中，该任务点将被自动跳过并打印警告。
    task_locations = [
        # B1（仅右臂，负 x 侧）
        {"x": -0.50, "y":  0.50, "region": "B1"},
        {"x": -0.40, "y":  0.40, "region": "B1"},
        {"x": -0.30, "y":  0.60, "region": "B1"},

        # B2（干涉区）
        {"x": -0.10, "y":  0.50, "region": "B2"},
        {"x":  0.00, "y":  0.40, "region": "B2"},
        {"x":  0.10, "y":  0.60, "region": "B2"},

        # B3（仅左臂，正 x 侧）
        {"x":  0.30, "y":  0.50, "region": "B3"},
        {"x":  0.40, "y":  0.40, "region": "B3"},

        # B4（仅右臂，负 x 侧）
        {"x": -0.50, "y": -0.50, "region": "B4"},
        {"x": -0.40, "y": -0.40, "region": "B4"},

        # B5（干涉区）
        {"x": -0.10, "y": -0.50, "region": "B5"},
        {"x":  0.00, "y": -0.40, "region": "B5"},
        {"x":  0.10, "y": -0.60, "region": "B5"},

        # B6（仅左臂，正 x 侧）
        {"x":  0.30, "y": -0.50, "region": "B6"},
        {"x":  0.40, "y": -0.40, "region": "B6"},
    ]

    task_df = planner.load_task_locations(task_locations)
    print(f"\n  加载了 {len(task_df)} 个有效任务点（不可达的点已跳过）")
    print_coverage_summary(task_df)

    print("\n  运行优化...")
    h_actions, m_actions, improvement = planner.solve_optimization(time_limit=20)

    h_makespan = max(a["end"] for a in h_actions)
    m_makespan = max(a["end"] for a in m_actions)
    print(f"\n  启发式 makespan : {h_makespan:.3f} s")
    print(f"  MILP makespan   : {m_makespan:.3f} s")
    print(f"  提升             : {improvement:.1f}%")

    result_dir = os.path.join(SCRIPT_DIR, "results", "realcost_example2")
    planner.save_results(result_dir)
    print(f"\n  结果已保存到 {result_dir}/")

    return planner


# =============================================================================
# 示例 3：对比 BaselinePlanner 与 RealCostPlanner 的 makespan 差异
# =============================================================================
def example_compare_baseline_vs_realcost():
    print("\n" + "=" * 60)
    print("RealCostPlanner 示例 3：与 BaselinePlanner 对比")
    print("=" * 60)

    if not os.path.exists(COST_TABLE_PATH):
        print(f"[错误] 未找到 cost table: {COST_TABLE_PATH}")
        return

    from dual_arm_planner import BaselinePlanner

    np.random.seed(42)

    points_per_region = {"B1": 4, "B2": 3, "B3": 3, "B4": 4, "B5": 3, "B6": 3}

    # ---- Baseline ----
    bp = BaselinePlanner()
    bp.create_task_dataset(points_per_region)
    bh, bm, _ = bp.solve_optimization(time_limit=10)
    b_heuristic = max(a["end"] for a in bh)
    b_milp = max(a["end"] for a in bm)

    # ---- RealCost ----
    np.random.seed(42)
    rcp = RealCostPlanner(cost_table_path=COST_TABLE_PATH)
    rcp.create_task_dataset(points_per_region)
    rh, rm, _ = rcp.solve_optimization(time_limit=10)
    r_heuristic = max(a["end"] for a in rh)
    r_milp = max(a["end"] for a in rm)

    print()
    print(f"  {'':20} {'Baseline':>12} {'RealCost':>12}")
    print(f"  {'-'*46}")
    print(f"  {'Heuristic makespan':20} {b_heuristic:>12.3f} {r_heuristic:>12.3f}")
    print(f"  {'MILP makespan':20} {b_milp:>12.3f} {r_milp:>12.3f}")
    print()
    print("  注：两者结果存在差异，因为：")
    print("    - BaselinePlanner 用欧氏距离近似移动时间")
    print("    - RealCostPlanner 用实际 IK 路径代价（来自 cost table）")
    print("    - 两者的任务点也不同（Baseline 连续采样，RealCost 离散网格采样）")


# =============================================================================
# main
# =============================================================================
if __name__ == "__main__":
    print("RealCostPlanner 使用示例")
    print("=" * 60)

    # 示例 1：随机采样 + 完整优化流程
    planner1 = example_basic()

    # 示例 2：手动指定坐标
    # planner2 = example_custom_locations()

    # 示例 3：与 BaselinePlanner 对比
    # example_compare_baseline_vs_realcost()

    print("\n所有示例完成！结果保存在 planning/results/ 目录下。")
