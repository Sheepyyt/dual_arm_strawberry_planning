#!/usr/bin/env python3
"""
check_trajectory_interference.py — 基于危险区的轨迹干涉检查
================================================================
对 ROI 表中每个可达目标点，利用 FK 计算 best solution 轨迹上的
关键点（肘点、腕点、末端点），检查是否经过由 compute_danger_zone.py
求解出的危险区 E_I。

关键点选取：
  - frame[4]：肘部/前臂区域（joint5 之后）
  - frame[6]：腕部区域（joint7 之后）
  - frame[8]：末端执行器（EE）

检查策略：
  只检查目标构型处的关键点位置。近端连杆（基座、肩部）在机械臂
  折叠态下必然处于危险区内，但它们靠近各自基座，不构成实际碰撞
  风险，因此排除在检查范围外。

运行前请先运行 compute_danger_zone.py 生成 danger_zone_grid.npz。

输出：roi/results/interference_labels.pkl
    {
        "<cost_table_key>": {
            "left_interference":  bool,   # 左臂到达该点的轨迹是否经过危险区
            "right_interference": bool,   # 右臂到达该点的轨迹是否经过危险区
        },
        ...
    }
"""

import os
import sys
import pickle

import numpy as np

# ---------------------------------------------------------------------------
# 路径设定
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(SCRIPT_DIR, "results")
ROI_TABLE_PATH = os.path.join(RESULT_DIR, "roi_table.pkl")
DANGER_ZONE_PATH = os.path.join(RESULT_DIR, "danger_zone_grid.npz")
OUTPUT_PATH = os.path.join(RESULT_DIR, "interference_labels.pkl")

URDF_PATH = os.path.join(SCRIPT_DIR, "..", "urdf", "dual_arm_ik_xy_centered.urdf")

# 关键帧索引（FK 输出中的帧编号）
# [4] = 肘部/前臂, [6] = 腕部, [8] = 末端执行器
KEY_FRAME_INDICES = [4, 6, 8]

# q_home（与 build_roi_table.py 中的 q_home 一致）
Q_HOME = np.array([0.10, -1.57, 0.23, -2.20, -1.40, -2.80, 0.0], dtype=float)


# ===========================================================================
# 复用 compute_danger_zone.py 中的类
# ===========================================================================
sys.path.insert(0, SCRIPT_DIR)
from compute_danger_zone import URDFArmChain, OccupancyGrid


def load_danger_zone_grid(path):
    """从 .npz 文件加载预计算的危险区网格。"""
    data = np.load(path)
    grid = OccupancyGrid(
        float(data["x_min"]),
        float(data["x_min"]) + int(data["nx"]) * float(data["res"]),
        float(data["y_min"]),
        float(data["y_min"]) + int(data["ny"]) * float(data["res"]),
        float(data["res"]),
    )
    grid.grid = data["grid"]
    return grid


def _point_in_danger_zone(grid_I, x, y):
    """检查 XY 平面上的一个点是否落入危险区网格。"""
    ix = int((x - grid_I.x_min) / grid_I.res)
    iy = int((y - grid_I.y_min) / grid_I.res)
    if 0 <= ix < grid_I.nx and 0 <= iy < grid_I.ny:
        return bool(grid_I.grid[ix, iy])
    return False


def check_target_key_points(chain, q_target, grid_I,
                            key_indices=KEY_FRAME_INDICES):
    """
    检查目标构型处关键点是否落入危险区。

    Parameters
    ----------
    chain : URDFArmChain
    q_target : array-like, shape (n_joints,)
    grid_I : OccupancyGrid — 危险区网格
    key_indices : list[int] — 要检查的 FK 帧索引

    Returns
    -------
    bool — True 表示至少一个关键点在危险区内
    """
    pts = chain.fk_all_frames(np.asarray(q_target, dtype=float))
    for idx in key_indices:
        if _point_in_danger_zone(grid_I, pts[idx][0], pts[idx][1]):
            return True
    return False


def main():
    print("=" * 60)
    print("Trajectory Interference Check")
    print("=" * 60)

    # ---- 加载 ROI 表 ----
    if not os.path.isfile(ROI_TABLE_PATH):
        raise FileNotFoundError(f"ROI table not found: {ROI_TABLE_PATH}")
    with open(ROI_TABLE_PATH, "rb") as f:
        roi_payload = pickle.load(f)
    roi_data = roi_payload["data"]
    print(f"Loaded ROI table: {len(roi_data)} points")

    # ---- 加载危险区网格 ----
    if not os.path.isfile(DANGER_ZONE_PATH):
        raise FileNotFoundError(
            f"Danger zone grid not found: {DANGER_ZONE_PATH}\n"
            "Please run compute_danger_zone.py first."
        )
    grid_I = load_danger_zone_grid(DANGER_ZONE_PATH)
    n_I = np.count_nonzero(grid_I.grid)
    print(f"Loaded danger zone grid: {grid_I.nx}x{grid_I.ny}, "
          f"{n_I} cells ({n_I * grid_I.res**2 * 1e4:.1f} cm²)")

    # ---- 加载 FK 链 ----
    chain_L = URDFArmChain(URDF_PATH, "vehicle_to_left_arm", "left_arm")
    chain_R = URDFArmChain(URDF_PATH, "vehicle_to_right_arm", "right_arm")

    print(f"Key frame indices: {KEY_FRAME_INDICES}")

    # ---- 遍历每个 ROI 点，检查轨迹干涉 ----
    labels = {}
    total = len(roi_data)
    report_every = max(1, total // 20)

    for idx, (key, entry) in enumerate(roi_data.items()):
        if idx % report_every == 0:
            print(f"  [{idx}/{total}] checking {key} ...", flush=True)

        left_interference = False
        right_interference = False

        left_sol = entry["left"].get("best_solution")
        if left_sol is not None:
            left_interference = check_target_key_points(
                chain_L, left_sol, grid_I)

        right_sol = entry["right"].get("best_solution")
        if right_sol is not None:
            right_interference = check_target_key_points(
                chain_R, right_sol, grid_I)

        labels[key] = {
            "left_interference": left_interference,
            "right_interference": right_interference,
        }

    # ---- 统计 ----
    n_left_interf = sum(1 for v in labels.values() if v["left_interference"])
    n_right_interf = sum(1 for v in labels.values() if v["right_interference"])
    n_either = sum(1 for v in labels.values()
                   if v["left_interference"] or v["right_interference"])
    print(f"\nResults:")
    print(f"  Left arm interference targets:  {n_left_interf} / {total}")
    print(f"  Right arm interference targets: {n_right_interf} / {total}")
    print(f"  Either arm interference:        {n_either} / {total}")
    print(f"  Non-interference:               {total - n_either} / {total}")

    # ---- 保存 ----
    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(labels, f)
    print(f"\nSaved interference labels to {OUTPUT_PATH}")
    print("Done ✓")


if __name__ == "__main__":
    main()
