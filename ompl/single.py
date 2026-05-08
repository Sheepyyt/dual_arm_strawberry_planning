#!/usr/bin/env python3
"""
single.py — OMPL 单点/单臂调试脚本
- 默认从 ../roi/results/ 读取 roi_table_selected_points.pkl 和 danger_zone_data.npz
- mode=auto: 有 safe 解则优先 safe；若点本身在危险区，则直接使用 fallback/free
- mode=safe: 只使用 safe 解
- mode=free: 只使用 fallback 解；若没有则退回全部 goal_solutions
- 可视化新增：
    1) 所有 candidate goal 的终点姿态 -> 淡灰色骨架
    2) best path 最终落到的终点姿态 -> 红色高亮骨架
"""
import os
import sys
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ompl import base as ob
from ompl import geometric as og

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
ROI_DIR = os.path.join(PROJECT_ROOT, "roi")
ROI_RESULTS_DIR = os.path.join(ROI_DIR, "results")
OMPL_RESULTS_DIR = os.path.join(SCRIPT_DIR, "results", "single")
os.makedirs(OMPL_RESULTS_DIR, exist_ok=True)
if ROI_DIR not in sys.path:
    sys.path.insert(0, ROI_DIR)

from build_roi_table import CONFIG as BASE_CONFIG, load_base_transforms_from_urdf
from compute_danger_zone import URDFArmChain


class DangerMask2D:
    def __init__(self, grid, extent, res):
        self.grid = grid.astype(bool)
        self.x_min, self.x_max, self.y_min, self.y_max = [float(v) for v in extent]
        self.res = float(res)
        self.nx, self.ny = self.grid.shape

    def point_in_mask(self, x, y):
        ix = int((x - self.x_min) / self.res)
        iy = int((y - self.y_min) / self.res)
        if ix < 0 or ix >= self.nx or iy < 0 or iy >= self.ny:
            return False
        return bool(self.grid[ix, iy])

    def segment_hits_mask(self, p0, p1, sample_step=None):
        sample_step = sample_step or (0.5 * self.res)
        p0 = np.asarray(p0, dtype=float)
        p1 = np.asarray(p1, dtype=float)
        d = np.linalg.norm(p1[:2] - p0[:2])
        n = max(2, int(np.ceil(d / sample_step)) + 1)
        for t in np.linspace(0.0, 1.0, n):
            p = (1 - t) * p0 + t * p1
            if self.point_in_mask(float(p[0]), float(p[1])):
                return True
        return False


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--roi-pkl", type=str, default=os.path.join(ROI_RESULTS_DIR, "points", "roi_table_selected_points.pkl"))
    p.add_argument("--danger-npz", type=str, default=os.path.join(ROI_RESULTS_DIR, "danger", "danger_zone_data.npz"))
    p.add_argument("--arm", choices=["L", "R"], required=True)
    p.add_argument("--key", type=str, required=True)
    p.add_argument("--mode", choices=["auto", "safe", "free"], default="auto")
    p.add_argument("--solve-time", type=float, default=1.0)
    p.add_argument("--num-trials", type=int, default=5)
    p.add_argument("--interpolate-count", type=int, default=100)
    p.add_argument("--resolution", type=float, default=0.01)
    p.add_argument("--planner", choices=["RRTConnect", "RRTstar"], default="RRTConnect")
    return p


def load_masks(npz_path):
    z = np.load(npz_path)
    extent = z["extent"]
    res = float(z["res"][0])
    return {
        "upper": DangerMask2D(z["grid_I_upper"], extent, res),
        "lower": DangerMask2D(z["grid_I_lower"], extent, res),
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


def check_state_valid(chain, mask_obj, q):
    pts = chain.fk_all_frames(q)
    for i in range(len(pts) - 1):
        if mask_obj is not None and mask_obj.segment_hits_mask(pts[i], pts[i + 1]):
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
        return check_state_valid(self.chain, self.mask_obj, q)


def choose_planner(si, planner_name):
    return og.RRTstar(si) if planner_name == "RRTstar" else og.RRTConnect(si)


class MultipleGoalStates(ob.GoalSampleableRegion):
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


def plan_one_trial(space, joint_limits, q_home, goals, chain, mask_obj, solve_time, resolution, planner_name, interpolate_count):
    ss = og.SimpleSetup(space)
    si = ss.getSpaceInformation()
    checker = CustomStateValidityChecker(si, chain, mask_obj, len(joint_limits))
    ss.setStateValidityChecker(checker)
    si.setStateValidityCheckingResolution(float(resolution))
    ss.setStartState(q_to_state(space, q_home))
    goal_states_list = [q_to_state(space, qg) for qg in goals]
    ss.setGoal(MultipleGoalStates(si, goal_states_list))
    ss.setPlanner(choose_planner(si, planner_name))
    ss.setup()
    solved = ss.solve(float(solve_time))
    if not solved:
        return None
    ss.simplifySolution()
    path = ss.getSolutionPath()
    path.interpolate(int(interpolate_count))
    return np.asarray([state_to_q(path.getState(i), len(joint_limits)) for i in range(path.getStateCount())], dtype=float)


def path_time(q_path):
    if q_path is None or len(q_path) < 2:
        return None
    total = 0.0
    for i in range(len(q_path) - 1):
        total += float(np.max(np.abs(q_path[i + 1] - q_path[i])))
    return total


def select_goals(arm_data, mode, point_in_danger):
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


def count_valid_goals(goals, chain, mask_obj):
    valid = []
    for q in goals:
        if check_state_valid(chain, mask_obj, q):
            valid.append(q)
    return valid


def nearest_goal_index(best_goal_q, candidate_goals):
    if best_goal_q is None or not candidate_goals:
        return None
    d = [float(np.linalg.norm(np.asarray(g) - np.asarray(best_goal_q))) for g in candidate_goals]
    return int(np.argmin(d))


def plot_result(out_png, key, arm, used_mode, point_xyz, masks_dict, chain, q_path, T_left, T_right, candidate_goals):
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

    # 2) sampled robot postures along the best path (time progression)
    best_goal_q = None
    if q_path is not None and len(q_path) > 0:
        best_goal_q = q_path[-1]
        idxs = np.linspace(0, len(q_path) - 1, min(8, len(q_path))).astype(int)
        cmap = plt.cm.viridis(np.linspace(0.12, 0.95, len(idxs)))
        for k, idx in enumerate(idxs):
            pts = chain.fk_all_frames(q_path[idx])
            label = "path snapshots" if k == 0 else None
            ax.plot(pts[:, 0], pts[:, 1], "-", lw=1.6, alpha=0.9, color=cmap[k], label=label, zorder=2)
            ax.plot(pts[-1, 0], pts[-1, 1], "o", ms=3.5, color=cmap[k], zorder=2)

    # 3) highlight final chosen goal pose
    if best_goal_q is not None:
        pts = chain.fk_all_frames(best_goal_q)
        ax.plot(pts[:, 0], pts[:, 1], "-", lw=2.8, alpha=0.95, color="crimson", label="best final pose", zorder=3)
        ax.plot(pts[-1, 0], pts[-1, 1], "o", ms=5, color="crimson", zorder=3)
        goal_idx = nearest_goal_index(best_goal_q, candidate_goals)
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

    # joint trajectory plot
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


def main():
    args = build_parser().parse_args()
    cfg = dict(BASE_CONFIG)
    with open(args.roi_pkl, "rb") as f:
        roi = pickle.load(f)

    item = roi["data"][args.key]
    point_xyz = item["vehicle_xyz"]
    half = item.get("half", "upper" if point_xyz[1] >= 0.25 else "lower")
    point_in_danger = bool(item["left"].get("point_in_danger", False) if args.arm == "L" else item["right"].get("point_in_danger", False))
    arm_key = "left" if args.arm == "L" else "right"
    arm_data = item[arm_key]

    masks = load_masks(args.danger_npz)
    vis_mask = masks[half]
    active_mask = vis_mask if args.mode in ("safe", "auto") else None
    used_goals, used_mode = select_goals(arm_data, args.mode, point_in_danger)
    if not used_goals:
        raise RuntimeError(f"No candidate goals for {args.arm} @ {args.key} under mode={args.mode}")

    urdf_path = cfg["urdf_path"]
    T_left, T_right = load_base_transforms_from_urdf(urdf_path, cfg["left_base_joint"], cfg["right_base_joint"])
    chain = URDFArmChain(urdf_path, cfg["left_base_joint"], "left_arm") if args.arm == "L" else URDFArmChain(urdf_path, cfg["right_base_joint"], "right_arm")
    q_home = np.asarray(cfg["q_seed"], dtype=float)
    space = build_space(chain.joint_limits)

    planner_mask = active_mask if used_mode == "safe" else None
    if used_mode == "safe":
        valid_goals = count_valid_goals(used_goals, chain, planner_mask)
        print(f"[goal filter] total={len(used_goals)} valid_safe={len(valid_goals)} point_in_danger={point_in_danger}", flush=True)
        used_goals = valid_goals
    else:
        print(f"[goal filter] total={len(used_goals)} mode={used_mode} point_in_danger={point_in_danger}", flush=True)

    if not used_goals:
        print("[result] no valid goal states before planning", flush=True)
        out_png = os.path.join(OMPL_RESULTS_DIR, f"ompl_test_{args.arm}_{args.key.replace('.', 'p')}_{used_mode}.png")
        plot_result(out_png, args.key, args.arm, used_mode, point_xyz, masks, chain, None, T_left, T_right, [])
        return

    best_path, best_cost = None, None
    for trial in range(args.num_trials):
        q_path = plan_one_trial(space, chain.joint_limits, q_home, used_goals, chain, planner_mask,
                                args.solve_time, args.resolution, args.planner, args.interpolate_count)
        c = path_time(q_path)
        print(f"[trial {trial+1}/{args.num_trials}] success={q_path is not None} cost={c}", flush=True)
        if q_path is not None and (best_cost is None or c < best_cost):
            best_cost, best_path = c, q_path

    out_png = os.path.join(OMPL_RESULTS_DIR, f"ompl_test_{args.arm}_{args.key.replace('.', 'p')}_{used_mode}.png")
    plot_result(out_png, args.key, args.arm, used_mode, point_xyz, masks, chain, best_path, T_left, T_right, used_goals)
    if best_path is None:
        print("[result] no path found", flush=True)
    else:
        print(f"[result] best path cost = {best_cost:.6f}", flush=True)


if __name__ == "__main__":
    main()
