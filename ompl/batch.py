#!/usr/bin/env python3
"""
batch.py

对 roi_table_selected_points.pkl 中的所有点，按“每个 IK 解单独规划”的方式批量运行 OMPL。
输出统一写入：
    ompl/results/batch/

与 v2 的区别：
- 同一 (key, arm) 下，先跑完所有 IK
- 只给“成功且 best_cost 最小”的那一个 IK 保存图片
- 其它成功 IK 不保存图，避免大量重复图片
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
BATCH_OUT_DIR = os.path.join(SCRIPT_DIR, "results", "batch")
FIG_DIR = os.path.join(BATCH_OUT_DIR, "figures")
os.makedirs(BATCH_OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
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
    p.add_argument("--arms", choices=["L", "R", "both"], default="both")
    p.add_argument("--mode", choices=["auto", "safe", "free"], default="auto")
    p.add_argument("--solve-time", type=float, default=1.0)
    p.add_argument("--num-trials", type=int, default=5)
    p.add_argument("--interpolate-count", type=int, default=100)
    p.add_argument("--resolution", type=float, default=0.01)
    p.add_argument("--planner", choices=["RRTConnect", "RRTstar"], default="RRTConnect")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--save-figures", action="store_true", help="为每个 (key, arm) 只保存一张最优成功 IK 的图")
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


def plan_one_trial(space, joint_limits, q_home, goal_q, chain, mask_obj, solve_time, resolution, planner_name, interpolate_count):
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


def path_time(q_path):
    if q_path is None or len(q_path) < 2:
        return None
    total = 0.0
    for i in range(len(q_path) - 1):
        total += float(np.max(np.abs(q_path[i + 1] - q_path[i])))
    return total


def pick_goal_pool(arm_data, mode, point_in_danger):
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


def plot_per_ik_result(out_png, key, arm, used_mode, point_xyz, masks_dict, chain, q_path, T_left, T_right, goal_q):
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


def main():
    args = build_parser().parse_args()
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
            "output_dir": BATCH_OUT_DIR,
            "figure_dir": FIG_DIR if args.save_figures else None,
            "figure_policy": "best_success_per_(key,arm)",
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
        half = item.get("half", "upper" if point_xyz[1] >= 0.25 else "lower")
        mask_for_half = masks[half]

        for arm in arms:
            arm_key = "left" if arm == "L" else "right"
            arm_data = item[arm_key]
            point_in_danger = bool(arm_data.get("point_in_danger", False))
            goal_pool, used_mode = pick_goal_pool(arm_data, args.mode, point_in_danger)

            planner_mask = mask_for_half if used_mode == "safe" else None
            filtered_goals = []
            for q in goal_pool:
                if planner_mask is None or check_state_valid(chains[arm], planner_mask, q):
                    filtered_goals.append(q)

            per_ik_results = []
            best_success_idx = None
            best_success_cost = None

            for ik_idx, goal_q in enumerate(filtered_goals):
                total_tasks += 1
                best_path = None
                best_cost = None
                for _ in range(args.num_trials):
                    q_path = plan_one_trial(
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

            # only save figure for the single best successful IK under the same (key, arm)
            if args.save_figures and best_success_idx is not None:
                rec = per_ik_results[best_success_idx]
                fig_name = f"{key.replace('.', 'p')}_{arm}_ik{best_success_idx}_{used_mode}_BEST.png"
                fig_path = os.path.join(FIG_DIR, fig_name)
                best_path_arr = np.asarray(rec["best_path"], dtype=float) if rec["best_path"] is not None else None
                goal_q_arr = np.asarray(rec["goal_q"], dtype=float) if rec["goal_q"] is not None else None
                plot_per_ik_result(fig_path, key, arm, used_mode, point_xyz, masks, chains[arm], best_path_arr, T_left, T_right, goal_q_arr)
                fig_relpath = os.path.relpath(fig_path, BATCH_OUT_DIR)
                rec["figure_relpath"] = fig_relpath
                rec["is_best_success_for_key_arm"] = True
                # update corresponding summary row
                # rows for this key-arm are appended consecutively at the end of summary_rows
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

    out_pkl = os.path.join(BATCH_OUT_DIR, "per_ik.pkl")
    out_csv = os.path.join(BATCH_OUT_DIR, "per_ik.csv")
    out_txt = os.path.join(BATCH_OUT_DIR, "summary.txt")

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
        f.write(f"output_dir: {BATCH_OUT_DIR}\n")
        f.write(f"figure_dir: {FIG_DIR if args.save_figures else '(not saved)'}\n")
        f.write(f"total arm-point-ik tasks: {total_tasks}\n")
        f.write(f"successful plans: {success_tasks}\n")
        f.write(f"safe plans: {safe_success_tasks}\n")
        f.write(f"fallback/free plans: {free_success_tasks}\n")
        f.write(f"failed plans: {failed_tasks}\n")
        f.write(f"saved figures: {saved_figures}\n")
        f.write("figure policy: one best successful IK per (key, arm)\n")

    print(f"Saved pkl to {out_pkl}", flush=True)
    print(f"Saved csv to {out_csv}", flush=True)
    print(f"Saved txt to {out_txt}", flush=True)
    if args.save_figures:
        print(f"Saved figures under {FIG_DIR}", flush=True)


if __name__ == "__main__":
    main()
