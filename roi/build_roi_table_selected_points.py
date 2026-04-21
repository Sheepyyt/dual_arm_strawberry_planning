
#!/usr/bin/env python3
"""
build_roi_table_selected_points.py

按用户指定的固定点集：
  y = ±0.3, ±0.4, ±0.5
  x = 0, ±0.1, ±0.2, ±0.3, ±0.4
  z = 0.56

对每个点做 rich-IK 搜索，并输出：
  - roi/results/roi_table_selected_points.pkl
  - roi/results/dual_arm_cost_selected_points.pkl
  - roi/results/selected_points_reachability.png

可视化配色沿用 build_roi_table.py / plot_roi_coverage.py：
  unreachable   -> #BB5F76
  left reachable-> #AFC4E4
  right reachable-> #BEE4C8
  both reachable-> #5B71B5
"""
import os
import sys
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
    classify_entry,
)

try:
    from tracikpy import TracIKSolver
except Exception as e:
    print("[error] tracikpy import failed:", e, flush=True)
    raise


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--timeout", type=float, default=0.05)
    p.add_argument("--step-yaw", type=float, default=0.25)
    p.add_argument("--yaw-min", type=float, default=-np.pi)
    p.add_argument("--yaw-max", type=float, default=np.pi)
    p.add_argument("--yaw-limit-per-point", type=int, default=13, help="最多尝试多少个 yaw")
    p.add_argument("--max-attempts-per-yaw", type=int, default=24)
    p.add_argument("--max-solutions-per-yaw", type=int, default=4)
    p.add_argument("--n-random-seed-perturb", type=int, default=8)
    p.add_argument("--dist-eps", type=float, default=0.001)
    p.add_argument("--verbose", action="store_true")
    return p


def selected_points():
    x_values = [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4]
    y_values = [-0.5, -0.4, -0.3, 0.3, 0.4, 0.5]
    z = 0.56
    pts = []
    for x in x_values:
        for y in y_values:
            pts.append((float(x), float(y), float(z)))
    return pts


def build_center_out_order(yaw_values):
    vals = np.unique(np.round(np.asarray(yaw_values, dtype=float), 6))
    center_idx = int(np.argmin(np.abs(vals)))
    center = float(vals[center_idx])
    others = [float(v) for i, v in enumerate(vals) if i != center_idx]
    others.sort(key=lambda y: (abs(y), y))
    return np.array([center] + others, dtype=float)


def perturb_seed(base, scales=(0.03, 0.15, 0.02, 0.2, 0.2, 0.2, 0.3)):
    base = np.asarray(base, dtype=float)
    noise = np.random.normal(size=base.shape) * np.asarray(scales, dtype=float)
    return base + noise


def dedup_solutions(solutions, atol=1e-3):
    uniq = []
    for q in solutions:
        q = np.asarray(q, dtype=float)
        if not any(np.allclose(q, u, atol=atol, rtol=0.0) for u in uniq):
            uniq.append(q)
    return uniq


def make_seed_bank(q_home, prev_success, n_random):
    bank = [np.asarray(q_home, dtype=float)]
    if prev_success is not None:
        bank.append(np.asarray(prev_success, dtype=float))

    # 手工偏置，帮助搜不同支解
    manual = [
        np.array([0, -2, 0.23, -2.5, -1.0, -3.0,  0.0]),
        np.array([0, -1.4, 0.23, -2.0, -0.6, -2.4,  0.8]),
        np.array([0, -2.6, 0.23, -2.8, -1.3, -3.2, -0.8]),
        np.array([0, -1.8, 0.23, -1.7, -1.8, -2.2,  1.5]),
    ]
    for q in manual:
        bank.append(q.astype(float))

    for _ in range(n_random):
        bank.append(perturb_seed(q_home))

    return bank


def solve_for_arm_point(solver, T_vehicle_from_arm, p_vehicle, cfg, args):
    T_arm_from_vehicle = np.linalg.inv(T_vehicle_from_arm)
    p_local_h = T_arm_from_vehicle @ np.array([p_vehicle[0], p_vehicle[1], p_vehicle[2], 1.0], dtype=float)
    x_l, y_l, z_l = p_local_h[:3]

    yaw_values = np.arange(args.yaw_min, args.yaw_max + 0.5 * args.step_yaw, args.step_yaw)
    yaw_values = build_center_out_order(yaw_values)[: args.yaw_limit_per_point]

    q_home = np.asarray(cfg["q_seed"], dtype=float)
    best_time = None
    best_q = None
    best_yaw = None
    prev_success = None

    entry = {
        "angles": [],
        "solutions": {},          # yaw_key -> list[q]
        "goal_solutions": [],     # all unique q
        "goal_solution_meta": [], # [{yaw, q}]
        "best_yaw": None,
        "best_solution": None,
        "best_time": None,
        "local_xyz": [float(x_l), float(y_l), float(z_l)],
    }

    for yaw_i, yaw in enumerate(yaw_values):
        yaw_solutions = []
        seed_bank = make_seed_bank(q_home, prev_success, args.n_random_seed_perturb)
        n_attempt = min(args.max_attempts_per_yaw, len(seed_bank))

        if args.verbose:
            print(f"      yaw[{yaw_i+1}/{len(yaw_values)}]={yaw:.3f}, attempts={n_attempt}", flush=True)

        T_target = np.eye(4)
        # keep roll/pitch same as old script
        from build_roi_table import pose_xyz_yaw_to_T
        T_target = pose_xyz_yaw_to_T(
            float(x_l), float(y_l), float(z_l),
            float(yaw), cfg["ee_target_roll"], cfg["ee_target_pitch"]
        )

        for attempt_idx in range(n_attempt):
            qinit = np.asarray(seed_bank[attempt_idx], dtype=float)
            q = solver.ik(
                ee_pose=T_target,
                qinit=qinit,
                bx=args.dist_eps, by=args.dist_eps, bz=args.dist_eps,
                brx=cfg["roll_eps"], bry=cfg["pitch_eps"], brz=cfg["yaw_eps"],
            )
            if q is None:
                continue
            yaw_solutions.append(np.asarray(q, dtype=float))
            if prev_success is None:
                prev_success = np.asarray(q, dtype=float)
            if len(yaw_solutions) >= args.max_solutions_per_yaw:
                break

        yaw_solutions = dedup_solutions(yaw_solutions)
        if yaw_solutions:
            yk = angle_key(yaw)
            entry["angles"].append(float(yaw))
            entry["solutions"][yk] = [q.tolist() for q in yaw_solutions]
            for q in yaw_solutions:
                entry["goal_solutions"].append(q.tolist())
                entry["goal_solution_meta"].append({"yaw": float(yaw), "q": q.tolist()})
                t = motion_time(q_home, q, joint_speed=cfg["joint_speed"])
                if best_time is None or t < best_time:
                    best_time = float(t)
                    best_q = q.copy()
                    best_yaw = float(yaw)

    entry["goal_solutions"] = [q.tolist() for q in dedup_solutions(entry["goal_solutions"])]
    entry["n_goal_solutions"] = len(entry["goal_solutions"])
    if best_q is not None:
        entry["best_solution"] = best_q.tolist()
        entry["best_yaw"] = best_yaw
        entry["best_time"] = best_time
        entry["classification"] = "reachable"
    else:
        entry["classification"] = "unreachable"

    return entry


def plot_reachability(roi_payload, out_png, T_left, T_right):
    target_z = float(roi_payload["meta"]["z"])
    data = roi_payload["data"]

    left_groups = {"unreachable": [], "reachable": []}
    right_groups = {"unreachable": [], "reachable": []}
    summary_groups = {"both_unreachable": [], "left_only": [], "right_only": [], "both_reachable": []}

    for key, item in data.items():
        x, y, z = item["vehicle_xyz"]
        if not np.isclose(z, target_z, atol=1e-6):
            continue
        cls_left = item["left"]["classification"]
        cls_right = item["right"]["classification"]

        left_groups[cls_left].append((x, y))
        right_groups[cls_right].append((x, y))

        left_ok = cls_left == "reachable"
        right_ok = cls_right == "reachable"
        if left_ok and right_ok:
            summary_groups["both_reachable"].append((x, y))
        elif left_ok and not right_ok:
            summary_groups["left_only"].append((x, y))
        elif (not left_ok) and right_ok:
            summary_groups["right_only"].append((x, y))
        else:
            summary_groups["both_unreachable"].append((x, y))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    def setup_ax(ax, title):
        veh = patches.Rectangle((-0.50, -0.25), 1.0, 0.50, linewidth=2, edgecolor="gray",
                                facecolor="lightgray", alpha=0.25, linestyle="--")
        ax.add_patch(veh)
        ax.scatter(T_left[0, 3], T_left[1, 3], c="#819CC4", marker="*", s=160, zorder=5, label="Left arm base")
        ax.scatter(T_right[0, 3], T_right[1, 3], c="#8CC19A", marker="*", s=160, zorder=5, label="Right arm base")
        # ROI boxes
        for roi, clr in [
            ({"x_min": -0.5, "x_max": 0.5, "y_min": 0.25, "y_max": 0.65}, "blue"),
            ({"x_min": -0.5, "x_max": 0.5, "y_min": -0.65, "y_max": -0.25}, "green"),
        ]:
            ax.add_patch(
                patches.Rectangle((roi["x_min"], roi["y_min"]),
                                  roi["x_max"] - roi["x_min"], roi["y_max"] - roi["y_min"],
                                  linewidth=1.5, edgecolor=clr, facecolor="none")
            )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(-0.48, 0.48)
        ax.set_ylim(-0.60, 0.60)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    setup_ax(axes[0], f"Left-arm IK coverage @ z={target_z:.2f}")
    for cls, pts in left_groups.items():
        if pts:
            pts = np.asarray(pts)
            axes[0].scatter(pts[:, 0], pts[:, 1], s=30, c={"unreachable":"#BB5F76","reachable":"#AFC4E4"}[cls], label=cls)
    axes[0].legend(loc="center left", fontsize=8)

    setup_ax(axes[1], f"Right-arm IK coverage @ z={target_z:.2f}")
    for cls, pts in right_groups.items():
        if pts:
            pts = np.asarray(pts)
            axes[1].scatter(pts[:, 0], pts[:, 1], s=30, c={"unreachable":"#BB5F76","reachable":"#BEE4C8"}[cls], label=cls)
    axes[1].legend(loc="center left", fontsize=8)

    setup_ax(axes[2], f"Combined reachability @ z={target_z:.2f}")
    cmap = {
        "both_unreachable": "#BB5F76",
        "left_only": "#AFC4E4",
        "right_only": "#BEE4C8",
        "both_reachable": "#5B71B5",
    }
    for cls, pts in summary_groups.items():
        if pts:
            pts = np.asarray(pts)
            axes[2].scatter(pts[:, 0], pts[:, 1], s=30, c=cmap[cls], label=cls)
    axes[2].legend(loc="center left", fontsize=8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=250, bbox_inches="tight")
    print(f"Saved figure to {out_png}", flush=True)


def main():
    args = build_parser().parse_args()
    cfg = dict(BASE_CONFIG)
    cfg["timeout"] = args.timeout
    cfg["step_yaw"] = args.step_yaw

    urdf_path = cfg["urdf_path"]
    out_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)

    roi_table_file = os.path.join(out_dir, "roi_table_selected_points.pkl")
    cost_file = os.path.join(out_dir, "dual_arm_cost_selected_points.pkl")
    fig_file = os.path.join(out_dir, "selected_points_reachability.png")

    T_left, T_right = load_base_transforms_from_urdf(
        urdf_path, cfg["left_base_joint"], cfg["right_base_joint"]
    )

    solver = TracIKSolver(
        urdf_file=urdf_path,
        base_link=cfg["ik_base_link"],
        tip_link=cfg["ik_tip_link"],
        timeout=args.timeout,
    )

    pts = selected_points()
    print(f"[info] total selected points = {len(pts)}", flush=True)

    payload = {
        "meta": {
            "source": "build_roi_table_selected_points.py",
            "z": 0.56,
            "x_values": [-0.4,-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3,0.4],
            "y_values": [-0.5,-0.4,-0.3,0.3,0.4,0.5],
            "args": vars(args),
        },
        "data": {},
    }
    cost_table = {}

    for idx, p in enumerate(pts, start=1):
        x, y, z = p
        key = create_key(x, y, z)
        print(f"[point {idx}/{len(pts)}] {key}", flush=True)
        left = solve_for_arm_point(solver, T_left, p, cfg, args)
        right = solve_for_arm_point(solver, T_right, p, cfg, args)
        payload["data"][key] = {
            "vehicle_xyz": [x, y, z],
            "left": left,
            "right": right,
        }
        cost_table[key] = {
            "left_cost": left.get("best_time"),
            "right_cost": right.get("best_time"),
            "left_reachable": left.get("classification") == "reachable",
            "right_reachable": right.get("classification") == "reachable",
        }

    with open(roi_table_file, "wb") as f:
        pickle.dump(payload, f, protocol=4)
    with open(cost_file, "wb") as f:
        pickle.dump(cost_table, f, protocol=4)

    print(f"Saved ROI table to {roi_table_file}", flush=True)
    print(f"Saved cost table to {cost_file}", flush=True)

    # stats
    left_reach = 0
    right_reach = 0
    dual_reach = 0
    for item in payload["data"].values():
        l = item["left"]["classification"] == "reachable"
        r = item["right"]["classification"] == "reachable"
        left_reach += int(l)
        right_reach += int(r)
        dual_reach += int(l and r)
    print({
        "left_reachable": left_reach,
        "right_reachable": right_reach,
        "dual_reachable": dual_reach,
    }, flush=True)

    plot_reachability(payload, fig_file, T_left, T_right)


if __name__ == "__main__":
    main()
