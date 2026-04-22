#!/usr/bin/env python3
"""
Batch OMPL planning over all points in roi_table_selected_points.pkl.

Suggested location in project:
  ompl/batch_plan_roi_selected_points.py

Reads:
  ../roi/results/roi_table_selected_points.pkl
  ../roi/results/danger_zone_data.npz

Writes:
  ompl/results/batch_plan_roi/motion_roi_table_selected_points.pkl
  ompl/results/batch_plan_roi/motion_roi_table_selected_points_summary.csv
  ompl/results/batch_plan_roi/batch_plan_summary.txt
  (optional) per-point figures when --save-figures is enabled
"""
import os
import sys
import csv
import pickle
import argparse
import numpy as np

from ompl import base as ob
from ompl import geometric as og

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
ROI_DIR = os.path.join(PROJECT_ROOT, "roi")
ROI_RESULTS_DIR = os.path.join(ROI_DIR, "results")
OMPL_RESULTS_DIR = os.path.join(SCRIPT_DIR, "results", "batch_plan_roi")
os.makedirs(OMPL_RESULTS_DIR, exist_ok=True)
if ROI_DIR not in sys.path:
    sys.path.insert(0, ROI_DIR)

from build_roi_table import CONFIG as BASE_CONFIG, load_base_transforms_from_urdf
from compute_danger_zone import URDFArmChain

# Reuse the user's current single-point debug script if present.
# Fallback to local helpers if not.
try:
    from test_ompl_single_arm_debug_auto import (
        DangerMask2D,
        build_space,
        check_state_valid,
        CustomStateValidityChecker,
        choose_planner,
        MultipleGoalStates,
        state_to_q,
        q_to_state,
        path_time,
        load_masks,
        select_goals,
        count_valid_goals,
        plot_result,
    )
except Exception:
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
        return [q for q in goals if check_state_valid(chain, mask_obj, q)]

    def plot_result(*args, **kwargs):
        return


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--roi-pkl", type=str, default=os.path.join(ROI_RESULTS_DIR, "roi_table_selected_points.pkl"))
    p.add_argument("--danger-npz", type=str, default=os.path.join(ROI_RESULTS_DIR, "danger_zone_data.npz"))
    p.add_argument("--mode", choices=["auto", "safe", "free"], default="auto")
    p.add_argument("--solve-time", type=float, default=1.0)
    p.add_argument("--num-trials", type=int, default=5)
    p.add_argument("--interpolate-count", type=int, default=100)
    p.add_argument("--resolution", type=float, default=0.01)
    p.add_argument("--planner", choices=["RRTConnect", "RRTstar"], default="RRTConnect")
    p.add_argument("--arms", choices=["L", "R", "both"], default="both")
    p.add_argument("--save-figures", action="store_true")
    p.add_argument("--figure-limit", type=int, default=0, help="0 means no limit when --save-figures is enabled")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p


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


def arm_entry(item, arm):
    return item["left"] if arm == "L" else item["right"]


def point_in_danger_for_arm(item, arm):
    return bool(arm_entry(item, arm).get("point_in_danger", False))


def prepare_chain_and_bases(cfg, arm):
    urdf_path = cfg["urdf_path"]
    T_left, T_right = load_base_transforms_from_urdf(urdf_path, cfg["left_base_joint"], cfg["right_base_joint"])
    if arm == "L":
        chain = URDFArmChain(urdf_path, cfg["left_base_joint"], "left_arm")
    else:
        chain = URDFArmChain(urdf_path, cfg["right_base_joint"], "right_arm")
    return chain, T_left, T_right


def summarize_label(used_mode, point_in_danger, arm_data, success):
    if not success:
        return "unplanned"
    if used_mode == "safe":
        return "safe"
    # used_mode == free
    return "fallback" if arm_data.get("fallback_goal_solutions") else "free"


def main():
    args = build_parser().parse_args()
    out_pkl = os.path.join(OMPL_RESULTS_DIR, "motion_roi_table_selected_points.pkl")
    out_csv = os.path.join(OMPL_RESULTS_DIR, "motion_roi_table_selected_points_summary.csv")
    out_txt = os.path.join(OMPL_RESULTS_DIR, "batch_plan_summary.txt")

    if os.path.exists(out_pkl) and not args.overwrite:
        raise SystemExit(f"Output already exists: {out_pkl}. Use --overwrite to replace it.")

    with open(args.roi_pkl, "rb") as f:
        roi = pickle.load(f)
    cfg = dict(BASE_CONFIG)
    masks = load_masks(args.danger_npz)
    q_home = np.asarray(cfg["q_seed"], dtype=float)

    arms = ["L", "R"] if args.arms == "both" else [args.arms]
    keys = list(roi["data"].keys())

    payload = {
        "meta": {
            "source_roi": args.roi_pkl,
            "danger_npz": args.danger_npz,
            "mode": args.mode,
            "solve_time": args.solve_time,
            "num_trials": args.num_trials,
            "interpolate_count": args.interpolate_count,
            "resolution": args.resolution,
            "planner": args.planner,
            "arms": arms,
        },
        "data": {},
    }

    summary_rows = []
    save_count = 0
    total_tasks = len(keys) * len(arms)
    done = 0

    for key in keys:
        item = roi["data"][key]
        point_xyz = item["vehicle_xyz"]
        half = item.get("half", "upper" if point_xyz[1] >= 0.25 else "lower")
        payload["data"][key] = {
            "vehicle_xyz": point_xyz,
            "half": half,
            "arms": {},
        }

        for arm in arms:
            done += 1
            arm_data = arm_entry(item, arm)
            point_in_danger = point_in_danger_for_arm(item, arm)
            used_goals, used_mode = select_goals(arm_data, args.mode, point_in_danger)
            chain, T_left, T_right = prepare_chain_and_bases(cfg, arm)
            space = build_space(chain.joint_limits)
            planner_mask = masks[half] if used_mode == "safe" else None

            if used_mode == "safe":
                used_goals = count_valid_goals(used_goals, chain, planner_mask)

            result = {
                "requested_mode": args.mode,
                "used_mode": used_mode,
                "point_in_danger": point_in_danger,
                "candidate_goal_count": int(len(used_goals)),
                "success": False,
                "best_cost": None,
                "best_path": None,
                "best_final_q": None,
                "label": "unplanned",
            }

            if args.verbose:
                print(f"[{done}/{total_tasks}] key={key} arm={arm} half={half} used_mode={used_mode} candidates={len(used_goals)}")

            if used_goals:
                best_path = None
                best_cost = None
                for trial in range(args.num_trials):
                    q_path = plan_one_trial(
                        space, chain.joint_limits, q_home, used_goals, chain, planner_mask,
                        args.solve_time, args.resolution, args.planner, args.interpolate_count,
                    )
                    c = path_time(q_path)
                    if args.verbose:
                        print(f"    trial {trial+1}/{args.num_trials}: success={q_path is not None} cost={c}")
                    if q_path is not None and (best_cost is None or c < best_cost):
                        best_cost = c
                        best_path = q_path

                if best_path is not None:
                    result["success"] = True
                    result["best_cost"] = float(best_cost)
                    result["best_path"] = best_path.tolist()
                    result["best_final_q"] = best_path[-1].tolist()
                    result["label"] = summarize_label(used_mode, point_in_danger, arm_data, True)

                    if args.save_figures and (args.figure_limit == 0 or save_count < args.figure_limit):
                        fig_path = os.path.join(OMPL_RESULTS_DIR, f"ompl_batch_{arm}_{key.replace('.', 'p')}_{used_mode}.png")
                        plot_result(fig_path, key, arm, used_mode, point_xyz, masks, chain, best_path, T_left, T_right, used_goals)
                        save_count += 1

            payload["data"][key]["arms"][arm] = result
            summary_rows.append({
                "key": key,
                "x": point_xyz[0],
                "y": point_xyz[1],
                "z": point_xyz[2],
                "half": half,
                "arm": arm,
                "requested_mode": args.mode,
                "used_mode": used_mode,
                "point_in_danger": point_in_danger,
                "candidate_goal_count": len(used_goals),
                "success": result["success"],
                "label": result["label"],
                "best_cost": result["best_cost"],
                "safe_goal_count": len(arm_data.get("safe_goal_solutions", [])),
                "fallback_goal_count": len(arm_data.get("fallback_goal_solutions", [])),
            })

    with open(out_pkl, "wb") as f:
        pickle.dump(payload, f, protocol=4)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()) if summary_rows else [])
        writer.writeheader()
        writer.writerows(summary_rows)

    total = len(summary_rows)
    success_n = sum(int(r["success"]) for r in summary_rows)
    safe_n = sum(int(r["label"] == "safe") for r in summary_rows)
    fallback_n = sum(int(r["label"] in ("fallback", "free")) for r in summary_rows)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"total arm-point tasks: {total}\n")
        f.write(f"successful plans: {success_n}\n")
        f.write(f"safe plans: {safe_n}\n")
        f.write(f"fallback/free plans: {fallback_n}\n")
        f.write(f"failed plans: {total - success_n}\n")

    print(f"Saved motion ROI table to {out_pkl}")
    print(f"Saved summary CSV to {out_csv}")
    print(f"Saved summary TXT to {out_txt}")
    print({
        "total_arm_point_tasks": total,
        "successful_plans": success_n,
        "safe_plans": safe_n,
        "fallback_or_free_plans": fallback_n,
        "failed_plans": total - success_n,
    })


if __name__ == "__main__":
    main()
