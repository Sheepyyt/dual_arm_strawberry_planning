#!/usr/bin/env python3
"""
build_roi_table_selected_points_cover_first_v2.py

在 cover_first 版本基础上新增：
1) 若 roi/results/roi_table_selected_points.pkl 已存在，则默认不重复求解，直接读取并重画图
2) 图中标注每个点左右臂的 safe/fallback 解数量：
   - 左臂: L s/f
   - 右臂: R s/f
3) 保留 --force-regenerate，可强制重新求解
"""
import os
import sys
import math
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from build_roi_table import (
    CONFIG as BASE_CONFIG,
    create_key,
    angle_key,
    motion_time,
    load_base_transforms_from_urdf,
    pose_xyz_yaw_to_T,
)
from compute_danger_zone import URDFArmChain

try:
    from tracikpy import TracIKSolver
except Exception as e:
    print("[error] tracikpy import failed:", e, flush=True)
    raise


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--timeout", type=float, default=0.035)
    p.add_argument("--dist-eps", type=float, default=0.001)
    p.add_argument("--yaw-min", type=float, default=-math.pi)
    p.add_argument("--yaw-max", type=float, default=math.pi)
    p.add_argument("--global-yaw-count", type=int, default=17)
    p.add_argument("--local-yaw-delta", type=float, default=0.12)
    p.add_argument("--enable-local-refine", action="store_true")
    p.add_argument("--n-global-random-seeds", type=int, default=8)
    p.add_argument("--n-home-perturb-seeds", type=int, default=4)
    p.add_argument("--max-attempts-per-yaw", type=int, default=12)
    p.add_argument("--max-solutions-per-yaw", type=int, default=3)
    p.add_argument("--max-safe-solutions", type=int, default=8)
    p.add_argument("--z", type=float, default=0.56)
    p.add_argument("--x-values", type=str, default="-0.4,-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3,0.4")
    p.add_argument("--y-values", type=str, default="-0.5,-0.4,-0.3,0.3,0.4,0.5")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force-regenerate", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p


def parse_float_list(s):
    return [float(v.strip()) for v in s.split(",") if v.strip()]


def selected_points(args):
    x_values = parse_float_list(args.x_values)
    y_values = parse_float_list(args.y_values)
    z = float(args.z)
    pts = []
    for x in x_values:
        for y in y_values:
            pts.append((float(x), float(y), z))
    return pts, x_values, y_values


def unique_list(values, atol=1e-9):
    out = []
    for v in values:
        if not any(abs(v - u) <= atol for u in out):
            out.append(v)
    return out


def build_global_yaw_list(args):
    vals = np.linspace(args.yaw_min, args.yaw_max, args.global_yaw_count, endpoint=True)
    return unique_list([float(v) for v in vals], atol=1e-8)


def build_refined_yaws(success_yaws, args):
    vals = []
    for y in success_yaws:
        vals.extend([y - args.local_yaw_delta, y + args.local_yaw_delta])
    vals = [max(args.yaw_min, min(args.yaw_max, v)) for v in vals]
    return unique_list(vals, atol=1e-8)


def dedup_solutions(solutions, atol=1e-3):
    uniq = []
    for q in solutions:
        q = np.asarray(q, dtype=float)
        if not any(np.allclose(q, u, atol=atol, rtol=0.0) for u in uniq):
            uniq.append(q)
    return uniq


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
        "grid_I_upper": z["grid_I_upper"],
        "grid_I_lower": z["grid_I_lower"],
        "extent": extent,
        "res": res,
    }


def choose_half(y):
    if y >= 0.25:
        return "upper"
    if y <= -0.25:
        return "lower"
    return "middle"


def point_in_corresponding_danger_mask(vehicle_xyz, masks):
    x, y, _ = vehicle_xyz
    half = choose_half(y)
    if half == "upper":
        mask = DangerMask2D(masks["grid_I_upper"], masks["extent"], masks["res"])
        return mask.point_in_mask(x, y), mask
    elif half == "lower":
        mask = DangerMask2D(masks["grid_I_lower"], masks["extent"], masks["res"])
        return mask.point_in_mask(x, y), mask
    return False, None


def check_state_valid(chain, mask_obj, q):
    pts = chain.fk_all_frames(q)
    for i in range(len(pts) - 1):
        if mask_obj is not None and mask_obj.segment_hits_mask(pts[i], pts[i + 1]):
            return False
    return True


def sample_global_seed(joint_limits, rng):
    q = []
    for lo, hi in joint_limits:
        q.append(rng.uniform(float(lo), float(hi)))
    return np.asarray(q, dtype=float)


def perturb_seed_large(base, rng, scales=(0.10, 0.70, 0.06, 0.90, 0.90, 1.10, 1.40)):
    base = np.asarray(base, dtype=float)
    noise = rng.normal(size=base.shape) * np.asarray(scales, dtype=float)
    return base + noise


def clip_to_limits(q, joint_limits):
    q = np.asarray(q, dtype=float).copy()
    for i, (lo, hi) in enumerate(joint_limits):
        q[i] = np.clip(q[i], float(lo), float(hi))
    return q


def make_seed_bank(q_home, joint_limits, rng, args):
    bank = [np.asarray(q_home, dtype=float)]
    manuals = [
        np.array([0.0, -2.4, 0.23, -2.8, -1.2, -3.0,  0.0]),
        np.array([0.0, -1.2, 0.23, -1.4, -2.4, -2.0,  1.8]),
        np.array([0.0, -2.8, 0.23, -3.0, -0.8,  2.8, -1.8]),
        np.array([0.0, -0.8, 0.23, -0.8, -2.8,  0.8,  2.5]),
        np.array([0.0, -1.8, 0.23, -2.2, -1.8,  1.8, -2.5]),
        np.array([0.0, -2.0, 0.23, -1.0, -0.5, -1.0,  0.5]),
    ]
    for q in manuals:
        bank.append(clip_to_limits(q, joint_limits))
    for _ in range(args.n_home_perturb_seeds):
        bank.append(clip_to_limits(perturb_seed_large(q_home, rng), joint_limits))
    for _ in range(args.n_global_random_seeds):
        bank.append(sample_global_seed(joint_limits, rng))
    out = []
    for q in bank:
        if not any(np.allclose(q, u, atol=1e-6, rtol=0.0) for u in out):
            out.append(q)
    return out


def try_yaw_with_seed_bank(solver, T_target, seed_bank, cfg, args):
    sols = []
    n_attempt = min(args.max_attempts_per_yaw, len(seed_bank))
    for i in range(n_attempt):
        qinit = np.asarray(seed_bank[i], dtype=float)
        q = solver.ik(
            ee_pose=T_target,
            qinit=qinit,
            bx=args.dist_eps, by=args.dist_eps, bz=args.dist_eps,
            brx=cfg["roll_eps"], bry=cfg["pitch_eps"], brz=cfg["yaw_eps"],
        )
        if q is None:
            continue
        sols.append(np.asarray(q, dtype=float))
        if len(sols) >= args.max_solutions_per_yaw:
            break
    return dedup_solutions(sols)


def solve_for_arm_point_cover_first(solver, chain, T_vehicle_from_arm, p_vehicle, cfg, args, masks, rng):
    T_arm_from_vehicle = np.linalg.inv(T_vehicle_from_arm)
    p_local_h = T_arm_from_vehicle @ np.array([p_vehicle[0], p_vehicle[1], p_vehicle[2], 1.0], dtype=float)
    x_l, y_l, z_l = p_local_h[:3]
    q_home = np.asarray(cfg["q_seed"], dtype=float)
    joint_limits = chain.joint_limits
    point_in_danger, safe_mask = point_in_corresponding_danger_mask(p_vehicle, masks)

    entry = {
        "angles": [], "solutions": {},
        "safe_goal_solutions": [], "fallback_goal_solutions": [],
        "goal_solutions": [], "goal_solution_meta": [],
        "best_yaw": None, "best_solution": None, "best_time": None,
        "local_xyz": [float(x_l), float(y_l), float(z_l)],
        "point_in_danger": bool(point_in_danger),
        "preferred_mode": None,
        "n_safe_goal_solutions": 0,
        "n_fallback_goal_solutions": 0,
        "classification": "unreachable",
    }

    seed_bank = make_seed_bank(q_home, joint_limits, rng, args)
    global_yaws = build_global_yaw_list(args)
    safe_bucket, fallback_bucket, success_yaws = [], [], []

    def process_yaws(yaw_values, stage_name):
        nonlocal safe_bucket, fallback_bucket, success_yaws
        for yaw in yaw_values:
            T_target = pose_xyz_yaw_to_T(float(x_l), float(y_l), float(z_l), float(yaw), cfg["ee_target_roll"], cfg["ee_target_pitch"])
            sols = try_yaw_with_seed_bank(solver, T_target, seed_bank, cfg, args)
            if args.verbose:
                print(f"      [{stage_name}] yaw={yaw:.3f} -> {len(sols)} sols", flush=True)
            if not sols:
                continue
            success_yaws.append(float(yaw))
            for q in sols:
                meta = {"yaw": float(yaw), "q": q.tolist()}
                if (not point_in_danger) and safe_mask is not None and check_state_valid(chain, safe_mask, q):
                    safe_bucket.append(meta)
                else:
                    fallback_bucket.append(meta)

    process_yaws(global_yaws, "global")
    if args.enable_local_refine and success_yaws:
        process_yaws(build_refined_yaws(success_yaws, args), "local")

    def dedup_meta(bucket):
        uniq = []
        for item in bucket:
            q = np.asarray(item["q"], dtype=float)
            if not any(np.allclose(q, np.asarray(u["q"], dtype=float), atol=1e-3, rtol=0.0) for u in uniq):
                uniq.append(item)
        return uniq

    safe_bucket = dedup_meta(safe_bucket)
    fallback_bucket = dedup_meta(fallback_bucket)
    entry["safe_goal_solutions"] = [b["q"] for b in safe_bucket[:args.max_safe_solutions]]
    entry["fallback_goal_solutions"] = [b["q"] for b in fallback_bucket]
    entry["n_safe_goal_solutions"] = len(entry["safe_goal_solutions"])
    entry["n_fallback_goal_solutions"] = len(entry["fallback_goal_solutions"])

    if entry["n_safe_goal_solutions"] > 0:
        final_bucket = safe_bucket[:args.max_safe_solutions]
        entry["preferred_mode"] = "safe"
    elif entry["n_fallback_goal_solutions"] > 0:
        final_bucket = fallback_bucket
        entry["preferred_mode"] = "fallback"
    else:
        final_bucket = []
        entry["preferred_mode"] = "none"

    by_yaw, best_time, best_q, best_yaw, final_goals = {}, None, None, None, []
    for item in final_bucket:
        yaw = float(item["yaw"])
        q = np.asarray(item["q"], dtype=float)
        by_yaw.setdefault(angle_key(yaw), []).append(q.tolist())
        final_goals.append(q.tolist())
        t = motion_time(q_home, q, joint_speed=cfg["joint_speed"])
        if best_time is None or t < best_time:
            best_time, best_q, best_yaw = float(t), q.copy(), yaw

    entry["solutions"] = by_yaw
    entry["angles"] = sorted(unique_list([float(item["yaw"]) for item in final_bucket]))
    entry["goal_solution_meta"] = [{"yaw": float(item["yaw"]), "q": item["q"]} for item in final_bucket]
    entry["goal_solutions"] = [q.tolist() for q in dedup_solutions(final_goals)] if final_goals else []
    entry["n_goal_solutions"] = len(entry["goal_solutions"])
    if best_q is not None:
        entry["best_solution"] = best_q.tolist()
        entry["best_yaw"] = best_yaw
        entry["best_time"] = best_time
        entry["classification"] = "reachable"
    return entry


def annotate_counts(ax, data, which="combined"):
    for item in data.values():
        x, y, _ = item["vehicle_xyz"]
        ls = int(item["left"].get("n_safe_goal_solutions", 0))
        lf = int(item["left"].get("n_fallback_goal_solutions", 0))
        rs = int(item["right"].get("n_safe_goal_solutions", 0))
        rf = int(item["right"].get("n_fallback_goal_solutions", 0))
        if which in ("left", "combined"):
            ax.text(x + 0.008, y + 0.012, f"L {ls}/{lf}", fontsize=6, color="#486A9A")
        if which in ("right", "combined"):
            dy = -0.012 if which == "combined" else 0.012
            ax.text(x + 0.008, y + dy, f"R {rs}/{rf}", fontsize=6, color="#4E8B61")


def plot_reachability(roi_payload, out_png, T_left, T_right):
    target_z = float(roi_payload["meta"]["z"])
    data = roi_payload["data"]
    left_groups = {"unreachable": [], "reachable": []}
    right_groups = {"unreachable": [], "reachable": []}
    summary_groups = {"both_unreachable": [], "left_only": [], "right_only": [], "both_reachable": []}
    for _, item in data.items():
        x, y, z = item["vehicle_xyz"]
        if not np.isclose(z, target_z, atol=1e-6):
            continue
        cls_left = item["left"]["classification"]
        cls_right = item["right"]["classification"]
        left_groups[cls_left].append((x, y))
        right_groups[cls_right].append((x, y))
        l = cls_left == "reachable"
        r = cls_right == "reachable"
        if l and r:
            summary_groups["both_reachable"].append((x, y))
        elif l and not r:
            summary_groups["left_only"].append((x, y))
        elif (not l) and r:
            summary_groups["right_only"].append((x, y))
        else:
            summary_groups["both_unreachable"].append((x, y))

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    def setup_ax(ax, title):
        veh = patches.Rectangle((-0.50, -0.25), 1.0, 0.50, linewidth=2, edgecolor="gray", facecolor="lightgray", alpha=0.25, linestyle="--")
        ax.add_patch(veh)
        ax.scatter(T_left[0, 3], T_left[1, 3], c="#819CC4", marker="*", s=160, zorder=5, label="Left arm base")
        ax.scatter(T_right[0, 3], T_right[1, 3], c="#8CC19A", marker="*", s=160, zorder=5, label="Right arm base")
        for roi, clr in [({"x_min": -0.5, "x_max": 0.5, "y_min": 0.25, "y_max": 0.65}, "dimgray"), ({"x_min": -0.5, "x_max": 0.5, "y_min": -0.65, "y_max": -0.25}, "dimgray")]:
            ax.add_patch(patches.Rectangle((roi["x_min"], roi["y_min"]), roi["x_max"] - roi["x_min"], roi["y_max"] - roi["y_min"], linewidth=0.8, edgecolor=clr, facecolor="none"))
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(-0.52, 0.52)
        ax.set_ylim(-0.64, 0.64)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    setup_ax(axes[0], f"Left-arm IK coverage @ z={target_z:.2f}")
    for cls, pts in left_groups.items():
        if pts:
            pts = np.asarray(pts)
            axes[0].scatter(pts[:, 0], pts[:, 1], s=34, c={"unreachable":"#BB5F76","reachable":"#AFC4E4"}[cls], label=cls)
    annotate_counts(axes[0], data, which="left")
    axes[0].legend(loc="center left", fontsize=8)

    setup_ax(axes[1], f"Right-arm IK coverage @ z={target_z:.2f}")
    for cls, pts in right_groups.items():
        if pts:
            pts = np.asarray(pts)
            axes[1].scatter(pts[:, 0], pts[:, 1], s=34, c={"unreachable":"#BB5F76","reachable":"#BEE4C8"}[cls], label=cls)
    annotate_counts(axes[1], data, which="right")
    axes[1].legend(loc="center left", fontsize=8)

    setup_ax(axes[2], f"Combined reachability @ z={target_z:.2f}")
    cmap = {"both_unreachable": "#BB5F76", "left_only": "#AFC4E4", "right_only": "#BEE4C8", "both_reachable": "#5B71B5"}
    for cls, pts in summary_groups.items():
        if pts:
            pts = np.asarray(pts)
            axes[2].scatter(pts[:, 0], pts[:, 1], s=34, c=cmap[cls], label=cls)
    annotate_counts(axes[2], data, which="combined")
    axes[2].legend(loc="center left", fontsize=8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=260, bbox_inches="tight")
    print(f"Saved figure to {out_png}", flush=True)


def summarize(payload):
    left_reach = right_reach = dual_reach = 0
    for item in payload["data"].values():
        l = item["left"]["classification"] == "reachable"
        r = item["right"]["classification"] == "reachable"
        left_reach += int(l)
        right_reach += int(r)
        dual_reach += int(l and r)
    return {"left_reachable": left_reach, "right_reachable": right_reach, "dual_reachable": dual_reach}


def main():
    args = build_parser().parse_args()
    cfg = dict(BASE_CONFIG)
    cfg["timeout"] = args.timeout
    urdf_path = cfg["urdf_path"]
    out_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)
    roi_table_file = os.path.join(out_dir, "roi_table_selected_points.pkl")
    fig_file = os.path.join(out_dir, "selected_points_reachability.png")
    danger_npz = os.path.join(out_dir, "danger_zone_data.npz")
    T_left, T_right = load_base_transforms_from_urdf(urdf_path, cfg["left_base_joint"], cfg["right_base_joint"])

    if os.path.exists(roi_table_file) and not args.force_regenerate:
        with open(roi_table_file, "rb") as f:
            payload = pickle.load(f)
        print(f"[info] existing ROI table found, skip solving: {roi_table_file}", flush=True)
        print(summarize(payload), flush=True)
        plot_reachability(payload, fig_file, T_left, T_right)
        return

    if not os.path.exists(danger_npz):
        raise FileNotFoundError(f"danger zone file not found: {danger_npz}")

    rng = np.random.default_rng(args.seed)
    masks = load_masks(danger_npz)
    solver = TracIKSolver(urdf_file=urdf_path, base_link=cfg["ik_base_link"], tip_link=cfg["ik_tip_link"], timeout=args.timeout)
    left_chain = URDFArmChain(urdf_path, cfg["left_base_joint"], "left_arm")
    right_chain = URDFArmChain(urdf_path, cfg["right_base_joint"], "right_arm")
    pts, x_values, y_values = selected_points(args)
    print(f"[info] total selected points = {len(pts)}", flush=True)

    payload = {"meta": {"source": os.path.basename(__file__), "z": float(args.z), "x_values": x_values, "y_values": y_values, "args": vars(args)}, "data": {}}
    for idx, p in enumerate(pts, start=1):
        key = create_key(*p)
        print(f"[point {idx}/{len(pts)}] {key}", flush=True)
        left = solve_for_arm_point_cover_first(solver, left_chain, T_left, p, cfg, args, masks, rng)
        right = solve_for_arm_point_cover_first(solver, right_chain, T_right, p, cfg, args, masks, rng)
        payload["data"][key] = {"vehicle_xyz": [float(p[0]), float(p[1]), float(p[2])], "left": left, "right": right}
        if args.verbose:
            print(f"    L: {left['classification']} | mode={left['preferred_mode']} | safe={left['n_safe_goal_solutions']} fallback={left['n_fallback_goal_solutions']} final={left.get('n_goal_solutions', 0)}", flush=True)
            print(f"    R: {right['classification']} | mode={right['preferred_mode']} | safe={right['n_safe_goal_solutions']} fallback={right['n_fallback_goal_solutions']} final={right.get('n_goal_solutions', 0)}", flush=True)

    with open(roi_table_file, "wb") as f:
        pickle.dump(payload, f, protocol=4)
    print(f"Saved ROI table to {roi_table_file}", flush=True)
    print(summarize(payload), flush=True)
    plot_reachability(payload, fig_file, T_left, T_right)


if __name__ == "__main__":
    main()
