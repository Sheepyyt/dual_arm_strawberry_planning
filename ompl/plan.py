#!/usr/bin/env python3
"""
plan.py —— OMPL 运动规划（单点调试 + 批量 per-IK，合二为一）

两个子命令：
  single : 对某一个点、某一只臂做单次规划并画图（调试用）
      python ompl/plan.py single --arm L --key 0.000_0.300_0.560 --mode auto
  batch  : 对所有点、所有臂、所有 IK 解逐个规划，产出优化器所需的 per-IK 数据
      python ompl/plan.py batch --mode auto --arms both --solve-time 1.0 \
          --num-trials 5 --planner RRTConnect --save-figures

要点：
- safe 判定时使用与 danger zone 构建一致的 LINK_RADIUS 厚度检查
- single 用多目标（一次可收敛到多个候选 IK 之一），batch 对每个 IK 单独规划

运行环境：ompl
"""

import os
import sys
import csv
import pickle
import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from ompl import base as ob
from ompl import geometric as og

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
ROI_DIR = os.path.join(PROJECT_ROOT, "roi")
ROI_RESULTS_DIR = os.path.join(ROI_DIR, "results")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if ROI_DIR not in sys.path:
    sys.path.insert(0, ROI_DIR)

from build_roi_table import CONFIG as BASE_CONFIG, load_base_transforms_from_urdf
from compute_danger_zone import URDFArmChain
from config import LINK_RADIUS


# ===========================================================
# 公共部分：危险区栅格 + OMPL 工具
# ===========================================================
class DangerMask2D:
    """危险区 2D 占用栅格 + 点/线段命中查询（带离散化边界容差）。"""

    def __init__(self, grid, extent, res):
        self.grid = grid.astype(bool)
        self.x_min, self.x_max, self.y_min, self.y_max = [float(v) for v in extent]
        self.res = float(res)
        self.nx, self.ny = self.grid.shape
        self.half_cell_diag = 0.5 * np.sqrt(2.0) * self.res

    def point_in_mask(self, x, y, margin=0.0):
        effective = float(max(0.0, margin)) + self.half_cell_diag
        ix_c = int((x - self.x_min) / self.res)
        iy_c = int((y - self.y_min) / self.res)
        r = int(np.ceil(effective / self.res))
        ix_lo = max(0, ix_c - r)
        ix_hi = min(self.nx - 1, ix_c + r)
        iy_lo = max(0, iy_c - r)
        iy_hi = min(self.ny - 1, iy_c + r)
        if ix_lo > ix_hi or iy_lo > iy_hi:
            return False
        for ix in range(ix_lo, ix_hi + 1):
            cx = self.x_min + (ix + 0.5) * self.res
            dx2 = (cx - x) ** 2
            if dx2 > effective * effective:
                continue
            for iy in range(iy_lo, iy_hi + 1):
                if not self.grid[ix, iy]:
                    continue
                cy = self.y_min + (iy + 0.5) * self.res
                if dx2 + (cy - y) ** 2 <= effective * effective:
                    return True
        return False

    def segment_hits_mask(self, p0, p1, query_radius=0.0, sample_step=None):
        sample_step = sample_step or (0.5 * self.res)
        p0 = np.asarray(p0, dtype=float)
        p1 = np.asarray(p1, dtype=float)
        d = np.linalg.norm(p1[:2] - p0[:2])
        n = max(2, int(np.ceil(d / sample_step)) + 1)
        for t in np.linspace(0.0, 1.0, n):
            p = (1 - t) * p0 + t * p1
            if self.point_in_mask(float(p[0]), float(p[1]), margin=query_radius):
                return True
        return False


def load_masks(npz_path):
    z = np.load(npz_path)
    extent = z["extent"]
    res = float(z["res"][0])
    return {
        "upper": DangerMask2D(z["grid_I_upper"], extent, res),
        "lower": DangerMask2D(z["grid_I_lower"], extent, res),
        "res": res,
    }


def state_to_q(state, dof):
    return np.array([state[i] for i in range(dof)], dtype=float)


def q_to_state(space, q):
    s = space.allocState()
    for i, v in enumerate(q):
        s[i] = float(v)
    return s


def build_space(joint_limits):
    n = len(joint_limits)
    space = ob.RealVectorStateSpace(n)
    bounds = ob.RealVectorBounds(n)
    for i, (lo, hi) in enumerate(joint_limits):
        bounds.setLow(i, float(lo))
        bounds.setHigh(i, float(hi))
    space.setBounds(bounds)
    return space


def check_state_valid(chain, mask_obj, q, link_radius=LINK_RADIUS):
    """逐节连杆做 XY 投影，检查是否扫过危险区；任一节命中即判为不合法。"""
    pts = chain.fk_all_frames(q)
    for i in range(len(pts) - 1):
        if mask_obj is not None and mask_obj.segment_hits_mask(pts[i], pts[i + 1], query_radius=link_radius):
            return False
    return True


class CustomStateValidityChecker(ob.StateValidityChecker):
    def __init__(self, si, chain, mask_obj, dof):
        super().__init__(si)
        self.si_ = si
        self.chain = chain
        self.mask_obj = mask_obj
        self.dof = dof

    def isValid(self, state):
        if not self.si_.satisfiesBounds(state):
            return False
        q = state_to_q(state, self.dof)
        return check_state_valid(self.chain, self.mask_obj, q, link_radius=LINK_RADIUS)


def choose_planner(si, planner_name):
    return og.RRTstar(si) if planner_name == "RRTstar" else og.RRTConnect(si)


def path_time(q_path):
    """路径代价：相邻两步“最大关节变化量”之和。"""
    if q_path is None or len(q_path) < 2:
        return None
    total = 0.0
    for i in range(len(q_path) - 1):
        total += float(np.max(np.abs(q_path[i + 1] - q_path[i])))
    return total


def pick_goal_pool(arm_data, mode, point_in_danger):
    """按 mode（safe/free/auto）和点是否在危险区，选出该臂要用的候选终点集合。"""
    safe_goals = [np.asarray(q, dtype=float) for q in arm_data.get("safe_goal_solutions", [])]
    fallback_goals = [np.asarray(q, dtype=float) for q in arm_data.get("fallback_goal_solutions", [])]
    all_goals = [np.asarray(q, dtype=float) for q in arm_data.get("goal_solutions", [])]

    if mode == "safe":
        return safe_goals, "safe"
    if mode == "free":
        return (fallback_goals or all_goals), "free"
    if point_in_danger:
        return (fallback_goals or all_goals), "free"
    if safe_goals:
        return safe_goals, "safe"
    return (fallback_goals or all_goals), "free"


class MultipleGoalStates(ob.GoalSampleableRegion):
    """多目标终点：一次规划可收敛到若干候选 IK 解中的任意一个（single 子命令用）。"""

    def __init__(self, si, states, threshold=1e-3):
        super().__init__(si)
        self.states = states
        self.setThreshold(threshold)

    def distanceGoal(self, state):
        dists = [self.getSpaceInformation().getStateSpace().distance(state, g) for g in self.states]
        return min(dists) if dists else float("inf")

    def sampleGoal(self, state):
        import random
        if self.states:
            g = random.choice(self.states)
            self.getSpaceInformation().getStateSpace().copyState(state, g)

    def maxSampleCount(self):
        return len(self.states)

    def couldSample(self):
        return len(self.states) > 0


# ===========================================================
# single 子命令：单点 / 单臂调试
# ===========================================================
def _plan_one_trial_multi(space, joint_limits, q_home, goals, chain, mask_obj, solve_time, resolution, planner_name, interpolate_count):
    ss = og.SimpleSetup(space)
    si = ss.getSpaceInformation()
    checker = CustomStateValidityChecker(si, chain, mask_obj, len(joint_limits))
    ss.setStateValidityChecker(checker)
    si.setStateValidityCheckingResolution(float(resolution))
    ss.setStartState(q_to_state(space, q_home))
    ss.setGoal(MultipleGoalStates(si, [q_to_state(space, qg) for qg in goals]))
    ss.setPlanner(choose_planner(si, planner_name))
    ss.setup()
    solved = ss.solve(float(solve_time))
    if not solved:
        return None
    ss.simplifySolution()
    path = ss.getSolutionPath()
    path.interpolate(int(interpolate_count))
    return np.asarray([state_to_q(path.getState(i), len(joint_limits)) for i in range(path.getStateCount())], dtype=float)


def _count_valid_goals(goals, chain, mask_obj):
    valid = []
    for q in goals:
        if check_state_valid(chain, mask_obj, q, link_radius=LINK_RADIUS):
            valid.append(q)
    return valid


def _nearest_goal_index(best_goal_q, candidate_goals):
    if best_goal_q is None or not candidate_goals:
        return None
    d = [float(np.linalg.norm(np.asarray(g) - np.asarray(best_goal_q))) for g in candidate_goals]
    return int(np.argmin(d))


def _plot_single(out_png, key, arm, used_mode, point_xyz, masks_dict, chain, q_path, T_left, T_right, candidate_goals):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    veh = patches.Rectangle((-0.50, -0.25), 1.0, 0.50, linewidth=2, edgecolor="gray", facecolor="lightgray", alpha=0.25, linestyle="--")
    ax.add_patch(veh)
    ax.scatter(T_left[0, 3], T_left[1, 3], c="#819CC4", marker="*", s=140, label="Left arm base")
    ax.scatter(T_right[0, 3], T_right[1, 3], c="#8CC19A", marker="*", s=140, label="Right arm base")
    ax.scatter([point_xyz[0]], [point_xyz[1]], c="red", marker="x", s=80, label="target")

    if masks_dict is not None:
        du = masks_dict["upper"]
        ix, iy = np.where(du.grid)
        xs = du.x_min + (ix + 0.5) * du.res
        ys = du.y_min + (iy + 0.5) * du.res
        ax.scatter(xs, ys, s=2, c="#e8a838", alpha=0.18, label="danger mask (upper)")
        dl = masks_dict["lower"]
        ix, iy = np.where(dl.grid)
        xs = dl.x_min + (ix + 0.5) * dl.res
        ys = dl.y_min + (iy + 0.5) * dl.res
        ax.scatter(xs, ys, s=2, c="#9b59b6", alpha=0.18, label="danger mask (lower)")

    if candidate_goals:
        for qg in candidate_goals:
            pts = chain.fk_all_frames(qg)
            ax.plot(pts[:, 0], pts[:, 1], "-", lw=1.0, alpha=0.18, color="gray")

    best_goal_q = None
    if q_path is not None and len(q_path) > 0:
        best_goal_q = q_path[-1]
        idxs = np.linspace(0, len(q_path) - 1, min(8, len(q_path))).astype(int)
        cmap = plt.cm.viridis(np.linspace(0.12, 0.95, len(idxs)))
        for k, idx in enumerate(idxs):
            pts = chain.fk_all_frames(q_path[idx])
            label = "path snapshots" if k == 0 else None
            ax.plot(pts[:, 0], pts[:, 1], "-", lw=1.6, alpha=0.9, color=cmap[k], label=label)
            ax.plot(pts[-1, 0], pts[-1, 1], "o", ms=3.5, color=cmap[k])

    if best_goal_q is not None:
        pts = chain.fk_all_frames(best_goal_q)
        ax.plot(pts[:, 0], pts[:, 1], "-", lw=2.8, alpha=0.95, color="crimson", label="best final pose")
        ax.plot(pts[-1, 0], pts[-1, 1], "o", ms=5, color="crimson")
        goal_idx = _nearest_goal_index(best_goal_q, candidate_goals)
        if goal_idx is not None:
            ax.text(pts[-1, 0] + 0.01, pts[-1, 1] + 0.01, f"goal#{goal_idx}", fontsize=8, color="crimson")

    ax.set_title(f"{arm} arm {used_mode} path overlay")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_xlim(-0.55, 0.55)
    ax.set_ylim(-0.7, 0.7)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    ax2 = axes[1]
    if q_path is not None:
        for j in range(q_path.shape[1]):
            ax2.plot(q_path[:, j], label=f"q{j+1}")
        ax2.legend(fontsize=7, ncol=2)
    ax2.set_title("Joint trajectory")
    ax2.set_xlabel("path step")
    ax2.set_ylabel("joint value")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"OMPL single-arm test | key={key} | arm={arm} | mode={used_mode}")
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    print(f"Saved figure to {out_png}", flush=True)
    plt.close(fig)


def cmd_single(args):
    out_dir = os.path.join(SCRIPT_DIR, "results", "single")
    os.makedirs(out_dir, exist_ok=True)
    cfg = dict(BASE_CONFIG)

    with open(args.roi_pkl, "rb") as f:
        roi = pickle.load(f)

    item = roi["data"][args.key]
    point_xyz = item["vehicle_xyz"]
    half = "upper" if point_xyz[1] >= 0.25 else "lower"
    arm_key = "left" if args.arm == "L" else "right"
    arm_data = item[arm_key]
    point_in_danger = bool(arm_data.get("point_in_danger", False))

    masks = load_masks(args.danger_npz)
    used_goals, used_mode = pick_goal_pool(arm_data, args.mode, point_in_danger)

    if not used_goals:
        raise RuntimeError(f"No candidate goals for {args.arm} @ {args.key} under mode={args.mode}")

    urdf_path = cfg["urdf_path"]
    T_left, T_right = load_base_transforms_from_urdf(urdf_path, cfg["left_base_joint"], cfg["right_base_joint"])
    chain = URDFArmChain(urdf_path, cfg["left_base_joint"], "left_arm") if args.arm == "L" else URDFArmChain(urdf_path, cfg["right_base_joint"], "right_arm")
    q_home = np.asarray(cfg["q_seed"], dtype=float)
    space = build_space(chain.joint_limits)

    planner_mask = masks[half] if used_mode == "safe" else None
    if used_mode == "safe":
        valid_goals = _count_valid_goals(used_goals, chain, planner_mask)
        print(f"[goal filter] total={len(used_goals)} valid_safe={len(valid_goals)} point_in_danger={point_in_danger}", flush=True)
        used_goals = valid_goals
    else:
        print(f"[goal filter] total={len(used_goals)} mode={used_mode} point_in_danger={point_in_danger}", flush=True)

    if not used_goals:
        print("[result] no valid goal states before planning", flush=True)
        out_png = os.path.join(out_dir, f"{args.arm}_{args.key.replace('.', 'p')}_{used_mode}.png")
        _plot_single(out_png, args.key, args.arm, used_mode, point_xyz, masks, chain, None, T_left, T_right, [])
        return

    best_path, best_cost = None, None
    for trial in range(args.num_trials):
        q_path = _plan_one_trial_multi(space, chain.joint_limits, q_home, used_goals, chain, planner_mask,
                                       args.solve_time, args.resolution, args.planner, args.interpolate_count)
        c = path_time(q_path)
        print(f"[trial {trial+1}/{args.num_trials}] success={q_path is not None} cost={c}", flush=True)
        if q_path is not None and (best_cost is None or c < best_cost):
            best_cost, best_path = c, q_path

    out_png = os.path.join(out_dir, f"{args.arm}_{args.key.replace('.', 'p')}_{used_mode}.png")
    _plot_single(out_png, args.key, args.arm, used_mode, point_xyz, masks, chain, best_path, T_left, T_right, used_goals)
    if best_path is None:
        print("[result] no path found", flush=True)
    else:
        print(f"[result] best path cost = {best_cost:.6f}", flush=True)


# ===========================================================
# batch 子命令：所有点 / 所有臂 / 所有 IK 解
# ===========================================================
def _plan_one_trial_single(space, joint_limits, q_home, goal_q, chain, mask_obj, solve_time, resolution, planner_name, interpolate_count):
    ss = og.SimpleSetup(space)
    si = ss.getSpaceInformation()
    checker = CustomStateValidityChecker(si, chain, mask_obj, len(joint_limits))
    ss.setStateValidityChecker(checker)
    si.setStateValidityCheckingResolution(float(resolution))
    ss.setStartState(q_to_state(space, q_home))
    ss.setGoalState(q_to_state(space, goal_q))
    ss.setPlanner(choose_planner(si, planner_name))
    ss.setup()
    solved = ss.solve(float(solve_time))
    if not solved:
        return None
    ss.simplifySolution()
    path = ss.getSolutionPath()
    path.interpolate(int(interpolate_count))
    return np.asarray([state_to_q(path.getState(i), len(joint_limits)) for i in range(path.getStateCount())], dtype=float)


def _plot_per_ik(out_png, key, arm, used_mode, point_xyz, masks_dict, chain, q_path, T_left, T_right, goal_q):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    veh = patches.Rectangle((-0.50, -0.25), 1.0, 0.50, linewidth=2, edgecolor="gray", facecolor="lightgray", alpha=0.25, linestyle="--")
    ax.add_patch(veh)
    ax.scatter(T_left[0, 3], T_left[1, 3], c="#819CC4", marker="*", s=140, label="Left arm base")
    ax.scatter(T_right[0, 3], T_right[1, 3], c="#8CC19A", marker="*", s=140, label="Right arm base")
    ax.scatter([point_xyz[0]], [point_xyz[1]], c="red", marker="x", s=80, label="target")

    if masks_dict is not None:
        du = masks_dict["upper"]
        ix, iy = np.where(du.grid)
        xs = du.x_min + (ix + 0.5) * du.res
        ys = du.y_min + (iy + 0.5) * du.res
        ax.scatter(xs, ys, s=2, c="#e8a838", alpha=0.18, label="danger mask (upper)")
        dl = masks_dict["lower"]
        ix, iy = np.where(dl.grid)
        xs = dl.x_min + (ix + 0.5) * dl.res
        ys = dl.y_min + (iy + 0.5) * dl.res
        ax.scatter(xs, ys, s=2, c="#9b59b6", alpha=0.18, label="danger mask (lower)")

    if goal_q is not None:
        pts = chain.fk_all_frames(goal_q)
        ax.plot(pts[:, 0], pts[:, 1], "-", lw=2.0, alpha=0.35, color="gray", label="goal pose")

    if q_path is not None and len(q_path) > 0:
        idxs = np.linspace(0, len(q_path) - 1, min(8, len(q_path))).astype(int)
        cmap = plt.cm.viridis(np.linspace(0.12, 0.95, len(idxs)))
        for k, idx in enumerate(idxs):
            pts = chain.fk_all_frames(q_path[idx])
            label = "path snapshots" if k == 0 else None
            ax.plot(pts[:, 0], pts[:, 1], "-", lw=1.6, alpha=0.9, color=cmap[k], label=label)
            ax.plot(pts[-1, 0], pts[-1, 1], "o", ms=3.5, color=cmap[k])
        pts = chain.fk_all_frames(q_path[-1])
        ax.plot(pts[:, 0], pts[:, 1], "-", lw=2.8, alpha=0.95, color="crimson", label="best final pose")
        ax.plot(pts[-1, 0], pts[-1, 1], "o", ms=5, color="crimson")

    ax.set_title(f"{arm} arm {used_mode} per-IK overlay")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_xlim(-0.55, 0.55)
    ax.set_ylim(-0.7, 0.7)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    ax2 = axes[1]
    if q_path is not None:
        for j in range(q_path.shape[1]):
            ax2.plot(q_path[:, j], label=f"q{j+1}")
        ax2.legend(fontsize=7, ncol=2)
    ax2.set_title("Joint trajectory")
    ax2.set_xlabel("path step")
    ax2.set_ylabel("joint value")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"OMPL per-IK batch | key={key} | arm={arm} | mode={used_mode}")
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def cmd_batch(args):
    batch_out_dir = os.path.join(SCRIPT_DIR, "results", "batch")
    fig_dir = os.path.join(batch_out_dir, "figures")
    os.makedirs(batch_out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    cfg = dict(BASE_CONFIG)

    with open(args.roi_pkl, "rb") as f:
        roi = pickle.load(f)

    masks = load_masks(args.danger_npz)
    urdf_path = cfg["urdf_path"]
    T_left, T_right = load_base_transforms_from_urdf(urdf_path, cfg["left_base_joint"], cfg["right_base_joint"])
    q_home = np.asarray(cfg["q_seed"], dtype=float)

    arms = ["L", "R"] if args.arms == "both" else [args.arms]
    chains = {
        "L": URDFArmChain(urdf_path, cfg["left_base_joint"], "left_arm"),
        "R": URDFArmChain(urdf_path, cfg["right_base_joint"], "right_arm"),
    }
    spaces = {a: build_space(chains[a].joint_limits) for a in arms}

    results = {
        "meta": {
            "requested_mode": args.mode,
            "arms": arms,
            "solve_time": args.solve_time,
            "num_trials": args.num_trials,
            "interpolate_count": args.interpolate_count,
            "resolution": args.resolution,
            "planner": args.planner,
            "output_dir": batch_out_dir,
            "figure_dir": fig_dir if args.save_figures else None,
            "figure_policy": "one best successful IK per (key, arm)",
            "safe_check_link_radius": float(LINK_RADIUS),
        },
        "data": defaultdict(dict),
    }

    summary_rows = []
    total_tasks = success_tasks = safe_success_tasks = free_success_tasks = failed_tasks = 0
    saved_figures = 0

    keys = sorted(roi["data"].keys())
    for key in keys:
        item = roi["data"][key]
        point_xyz = item["vehicle_xyz"]
        half = "upper" if point_xyz[1] >= 0.25 else "lower"
        mask_for_half = masks[half]

        for arm in arms:
            arm_key = "left" if arm == "L" else "right"
            arm_data = item[arm_key]
            point_in_danger = bool(arm_data.get("point_in_danger", False))
            goal_pool, used_mode = pick_goal_pool(arm_data, args.mode, point_in_danger)

            planner_mask = mask_for_half if used_mode == "safe" else None
            filtered_goals = []
            for q in goal_pool:
                if planner_mask is None or check_state_valid(chains[arm], planner_mask, q, link_radius=LINK_RADIUS):
                    filtered_goals.append(q)

            per_ik_results = []
            best_success_idx = None
            best_success_cost = None

            for ik_idx, goal_q in enumerate(filtered_goals):
                total_tasks += 1
                best_path = None
                best_cost = None
                for _ in range(args.num_trials):
                    q_path = _plan_one_trial_single(
                        spaces[arm],
                        chains[arm].joint_limits,
                        q_home,
                        goal_q,
                        chains[arm],
                        planner_mask,
                        args.solve_time,
                        args.resolution,
                        args.planner,
                        args.interpolate_count,
                    )
                    c = path_time(q_path)
                    if q_path is not None and (best_cost is None or c < best_cost):
                        best_cost = c
                        best_path = q_path

                success = best_path is not None
                if success:
                    success_tasks += 1
                    if used_mode == "safe":
                        safe_success_tasks += 1
                    else:
                        free_success_tasks += 1
                    if best_success_cost is None or best_cost < best_success_cost:
                        best_success_cost = best_cost
                        best_success_idx = ik_idx
                else:
                    failed_tasks += 1

                per_ik_results.append({
                    "ik_index": ik_idx,
                    "success": success,
                    "best_cost": best_cost,
                    "best_path": None if best_path is None else best_path.tolist(),
                    "goal_q": goal_q.tolist(),
                    "used_mode": used_mode,
                    "point_in_danger": point_in_danger,
                    "label": used_mode if success else "unplanned",
                    "figure_relpath": "",
                    "is_best_success_for_key_arm": False,
                })

                summary_rows.append({
                    "key": key,
                    "arm": arm,
                    "half": half,
                    "ik_index": ik_idx,
                    "requested_mode": args.mode,
                    "used_mode": used_mode,
                    "point_in_danger": point_in_danger,
                    "success": success,
                    "best_cost": "" if best_cost is None else best_cost,
                    "x": point_xyz[0],
                    "y": point_xyz[1],
                    "z": point_xyz[2],
                    "is_best_success_for_key_arm": False,
                    "figure_relpath": "",
                })

                if args.verbose:
                    print(f"[{key}] arm={arm} ik={ik_idx} mode={used_mode} success={success} cost={best_cost}", flush=True)

            if args.save_figures and best_success_idx is not None:
                rec = per_ik_results[best_success_idx]
                fig_name = f"{key.replace('.', 'p')}_{arm}_ik{best_success_idx}_{used_mode}_BEST.png"
                fig_path = os.path.join(fig_dir, fig_name)
                best_path_arr = np.asarray(rec["best_path"], dtype=float) if rec["best_path"] is not None else None
                goal_q_arr = np.asarray(rec["goal_q"], dtype=float) if rec["goal_q"] is not None else None
                _plot_per_ik(fig_path, key, arm, used_mode, point_xyz, masks, chains[arm], best_path_arr, T_left, T_right, goal_q_arr)
                fig_relpath = os.path.relpath(fig_path, batch_out_dir)
                rec["figure_relpath"] = fig_relpath
                rec["is_best_success_for_key_arm"] = True
                for row in reversed(summary_rows):
                    if row["key"] == key and row["arm"] == arm and row["ik_index"] == best_success_idx:
                        row["figure_relpath"] = fig_relpath
                        row["is_best_success_for_key_arm"] = True
                        break
                saved_figures += 1

            results["data"][key][arm] = {
                "half": half,
                "requested_mode": args.mode,
                "used_mode": used_mode,
                "point_in_danger": point_in_danger,
                "candidate_goal_count_before_filter": len(goal_pool),
                "candidate_goal_count_after_filter": len(filtered_goals),
                "best_success_ik_index": best_success_idx,
                "best_success_cost": best_success_cost,
                "per_ik_results": per_ik_results,
            }

    out_pkl = os.path.join(batch_out_dir, "per_ik.pkl")
    out_csv = os.path.join(batch_out_dir, "per_ik.csv")
    out_txt = os.path.join(batch_out_dir, "summary.txt")

    with open(out_pkl, "wb") as f:
        pickle.dump(results, f, protocol=4)

    fieldnames = [
        "key", "arm", "half", "ik_index", "requested_mode", "used_mode",
        "point_in_danger", "success", "best_cost", "x", "y", "z",
        "is_best_success_for_key_arm", "figure_relpath"
    ]
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"output_dir: {batch_out_dir}\n")
        f.write(f"figure_dir: {fig_dir if args.save_figures else '(not saved)'}\n")
        f.write(f"total arm-point-ik tasks: {total_tasks}\n")
        f.write(f"successful plans: {success_tasks}\n")
        f.write(f"safe plans: {safe_success_tasks}\n")
        f.write(f"fallback/free plans: {free_success_tasks}\n")
        f.write(f"failed plans: {failed_tasks}\n")
        f.write(f"saved figures: {saved_figures}\n")
        f.write("figure policy: one best successful IK per (key, arm)\n")
        f.write(f"safe_check_link_radius: {LINK_RADIUS}\n")

    print(f"Saved pkl to {out_pkl}", flush=True)
    print(f"Saved csv to {out_csv}", flush=True)
    print(f"Saved txt to {out_txt}", flush=True)
    if args.save_figures:
        print(f"Saved figures under {fig_dir}", flush=True)


# ===========================================================
# 命令行
# ===========================================================
def build_parser():
    p = argparse.ArgumentParser(description="OMPL 运动规划：single（单点调试）/ batch（批量 per-IK）")
    sub = p.add_subparsers(dest="cmd", required=True)

    base = argparse.ArgumentParser(add_help=False)
    base.add_argument("--roi-pkl", type=str, default=os.path.join(ROI_RESULTS_DIR, "points", "roi_table_selected_points.pkl"))
    base.add_argument("--danger-npz", type=str, default=os.path.join(ROI_RESULTS_DIR, "danger", "danger_zone_data.npz"))
    base.add_argument("--solve-time", type=float, default=1.0)
    base.add_argument("--num-trials", type=int, default=5)
    base.add_argument("--interpolate-count", type=int, default=100)
    base.add_argument("--resolution", type=float, default=0.01)
    base.add_argument("--planner", choices=["RRTConnect", "RRTstar"], default="RRTConnect")

    ps = sub.add_parser("single", parents=[base], help="单点 / 单臂调试")
    ps.add_argument("--arm", choices=["L", "R"], required=True)
    ps.add_argument("--key", type=str, required=True)
    ps.add_argument("--mode", choices=["auto", "safe", "free"], default="auto")
    ps.set_defaults(func=cmd_single)

    pb = sub.add_parser("batch", parents=[base], help="批量 per-IK 规划")
    pb.add_argument("--arms", choices=["L", "R", "both"], default="both")
    pb.add_argument("--mode", choices=["auto", "safe", "free"], default="auto")
    pb.add_argument("--verbose", action="store_true")
    pb.add_argument("--save-figures", action="store_true", help="为每个 (key, arm) 仅保存一张最优成功 IK 的图")
    pb.set_defaults(func=cmd_batch)

    return p


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
