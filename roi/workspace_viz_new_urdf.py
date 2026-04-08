#!/usr/bin/env python3
"""
workspace_viz_new_urdf.py
=========================
基于 **新构型** ``urdf/berrybolter_arm.urdf`` 的 FK 可达空间可视化。

该 URDF 是单臂描述（含 meshes），左右臂结构完全相同，
双臂安装位姿沿用 ``dual_arm_ik_xy_centered.urdf`` 中的
``vehicle_to_left_arm`` / ``vehicle_to_right_arm`` 固定关节。

脚本只对 **单臂** 做 FK 采样，另一臂结果通过安装位姿平移/旋转得到。

输出两张图（每张 3 个子图：左臂 / 右臂 / 交集）：
1. 末端执行器（EE）二维可达范围 + 双臂交集
2. 全部连杆/关节的二维扫掠范围 + 双臂交集

同时生成凸包版和 alpha-shape 凹边界版，共 4 张 PNG。
"""

import os
import sys
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon as MplPolygon
from scipy.spatial import ConvexHull

# ────────────────────────────────────────────
# paths
# ────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
NEW_URDF_PATH = os.path.join(_HERE, "..", "urdf", "berrybolter_arm.urdf")
OLD_URDF_PATH = os.path.join(_HERE, "..", "urdf", "dual_arm_ik_xy_centered.urdf")
OUT_DIR = os.path.join(_HERE, "results")
os.makedirs(OUT_DIR, exist_ok=True)

# ────────────────────────────────────────────
# Dual-arm mounting transforms
# (from the old URDF, where both arms are installed on a vehicle)
# ────────────────────────────────────────────
#   vehicle_to_left_arm:  xyz="-0.28336 -0.22875 0.3126"  rpy="0 0 0"
#   vehicle_to_right_arm: xyz=" 0.28336  0.22875 0.27926" rpy="0 0 -pi"

LEFT_ARM_MOUNT_XYZ = [-0.28336, -0.22875, 0.3126]
LEFT_ARM_MOUNT_RPY = [0.0, 0.0, 0.0]
RIGHT_ARM_MOUNT_XYZ = [0.28336, 0.22875, 0.27926]
RIGHT_ARM_MOUNT_RPY = [0.0, 0.0, -np.pi]

# ────────────────────────────────────────────
# basic transforms (same as old script)
# ────────────────────────────────────────────

def rpy_to_rot(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def xyzrpy_to_T(xyz, rpy):
    T = np.eye(4)
    T[:3, :3] = rpy_to_rot(rpy[0], rpy[1], rpy[2])
    T[:3, 3] = np.asarray(xyz, dtype=float)
    return T


# ────────────────────────────────────────────
# URDF chain parser (for single arm berrybolter_arm.urdf)
# ────────────────────────────────────────────

class JointInfo:
    def __init__(self, name, jtype, origin_T, axis, lower, upper, parent, child):
        self.name = name
        self.jtype = jtype
        self.origin_T = origin_T
        self.axis = axis
        self.lower = lower
        self.upper = upper
        self.parent = parent
        self.child = child


def parse_urdf_chain(urdf_path, base_link, tip_link):
    """
    解析 URDF 中从 base_link 到 tip_link 的运动链。
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    joints_by_child = {}
    for j in root.findall("joint"):
        child = j.find("child").attrib["link"]
        joints_by_child[child] = j

    chain_joints_xml = []
    current = tip_link
    while current != base_link:
        if current not in joints_by_child:
            raise ValueError(f"Cannot trace chain from {tip_link} to {base_link}: "
                             f"link '{current}' has no parent joint")
        j = joints_by_child[current]
        chain_joints_xml.append(j)
        current = j.find("parent").attrib["link"]
    chain_joints_xml.reverse()

    chain = []
    for j in chain_joints_xml:
        name = j.attrib["name"]
        jtype = j.attrib.get("type", "fixed")
        origin_el = j.find("origin")
        if origin_el is not None:
            xyz = [float(v) for v in origin_el.attrib.get("xyz", "0 0 0").split()]
            rpy = [float(v) for v in origin_el.attrib.get("rpy", "0 0 0").split()]
        else:
            xyz, rpy = [0, 0, 0], [0, 0, 0]
        axis_el = j.find("axis")
        if axis_el is not None:
            axis = np.array([float(v) for v in axis_el.attrib.get("xyz", "0 0 1").split()])
        else:
            axis = np.array([0.0, 0.0, 1.0])
        limit_el = j.find("limit")
        lower, upper = 0.0, 0.0
        if limit_el is not None:
            lower = float(limit_el.attrib.get("lower", "0"))
            upper = float(limit_el.attrib.get("upper", "0"))

        parent = j.find("parent").attrib["link"]
        child = j.find("child").attrib["link"]
        T = xyzrpy_to_T(xyz, rpy)
        chain.append(JointInfo(name, jtype, T, axis, lower, upper, parent, child))

    return chain


def rot_axis(axis, angle):
    axis = axis / (np.linalg.norm(axis) + 1e-15)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def fk_all_link_origins(chain, q_vec):
    T = np.eye(4)
    results = []
    qi = 0
    for joint in chain:
        T = T @ joint.origin_T
        if joint.jtype == "revolute":
            angle = q_vec[qi]
            R = rot_axis(joint.axis, angle)
            T_rot = np.eye(4)
            T_rot[:3, :3] = R
            T = T @ T_rot
            qi += 1
        elif joint.jtype == "prismatic":
            disp = q_vec[qi]
            T_disp = np.eye(4)
            T_disp[:3, 3] = joint.axis * disp
            T = T @ T_disp
            qi += 1
        results.append((joint.child, T.copy()))
    return results


def get_actuated_limits(chain):
    lowers, uppers = [], []
    for j in chain:
        if j.jtype in ("revolute", "prismatic"):
            lowers.append(j.lower)
            uppers.append(j.upper)
    return np.array(lowers), np.array(uppers)


# ────────────────────────────────────────────
# sampling
# ────────────────────────────────────────────

def sample_joint_configs(chain, n_samples=80000, mode="random"):
    lower, upper = get_actuated_limits(chain)
    ndof = len(lower)

    configs = []
    if mode in ("random", "both"):
        rng = np.random.default_rng(42)
        for _ in range(n_samples):
            q = lower + (upper - lower) * rng.random(ndof)
            configs.append(q)

    if mode in ("boundary", "both"):
        from itertools import product
        for combo in product(*[(lo, hi) for lo, hi in zip(lower, upper)]):
            configs.append(np.array(combo))

        mid = 0.5 * (lower + upper)
        for i in range(ndof):
            for val in [lower[i], upper[i]]:
                q = mid.copy()
                q[i] = val
                configs.append(q)

    return configs


# ────────────────────────────────────────────
# workspace computation
# ────────────────────────────────────────────

def compute_workspace(chain, n_samples=80000):
    configs = sample_joint_configs(chain, n_samples, mode="both")

    ee_points = []
    all_points = []
    link_points_dict = {}

    for q in configs:
        frames = fk_all_link_origins(chain, q)
        for lname, T in frames:
            xy = T[:2, 3]
            all_points.append(xy)
            if lname not in link_points_dict:
                link_points_dict[lname] = []
            link_points_dict[lname].append(xy)

        ee_T = frames[-1][1]
        ee_points.append(ee_T[:2, 3])

    ee_points = np.array(ee_points)
    all_points = np.array(all_points)
    for k in link_points_dict:
        link_points_dict[k] = np.array(link_points_dict[k])

    return ee_points, all_points, link_points_dict


# ────────────────────────────────────────────
# transform points
# ────────────────────────────────────────────

def transform_points_2d(points_xy, T_vehicle_from_arm):
    N = len(points_xy)
    ones = np.ones((N, 1))
    zeros = np.zeros((N, 1))
    pts_h = np.hstack([points_xy, zeros, ones])
    pts_vehicle = (T_vehicle_from_arm @ pts_h.T).T
    return pts_vehicle[:, :2]


# ────────────────────────────────────────────
# hull / shape helpers
# ────────────────────────────────────────────

def safe_convex_hull(pts):
    if len(pts) < 3:
        return None
    try:
        return ConvexHull(pts)
    except Exception:
        return None


def hull_polygon_xy(hull):
    if hull is None:
        return np.empty((0, 2))
    return hull.points[hull.vertices]


def _downsample_for_alpha(pts, max_pts=5000):
    if len(pts) <= max_pts:
        return pts
    rng = np.random.default_rng(0)
    idx = rng.choice(len(pts), max_pts, replace=False)
    return pts[idx]


def compute_alpha_shape(pts, alpha=0.0):
    if alpha == 0:
        hull = safe_convex_hull(pts)
        return hull_polygon_xy(hull) if hull else np.empty((0, 2))
    try:
        import alphashape as ash
        from shapely.geometry import MultiPolygon
        pts_ds = _downsample_for_alpha(pts, max_pts=3000)
        shape = ash.alphashape(pts_ds, alpha)
        if shape is None or shape.is_empty:
            hull = safe_convex_hull(pts)
            return hull_polygon_xy(hull) if hull else np.empty((0, 2))
        if isinstance(shape, MultiPolygon):
            largest = max(shape.geoms, key=lambda g: g.area)
            return np.array(largest.exterior.coords)
        if hasattr(shape, 'exterior'):
            return np.array(shape.exterior.coords)
        hull = safe_convex_hull(pts)
        return hull_polygon_xy(hull) if hull else np.empty((0, 2))
    except Exception:
        hull = safe_convex_hull(pts)
        return hull_polygon_xy(hull) if hull else np.empty((0, 2))


def compute_intersection_polygon(verts_a, verts_b):
    from shapely.geometry import Polygon as ShapelyPolygon, MultiPolygon
    if len(verts_a) < 3 or len(verts_b) < 3:
        return None, 0.0
    pa = ShapelyPolygon(verts_a)
    pb = ShapelyPolygon(verts_b)
    if not pa.is_valid:
        pa = pa.buffer(0)
    if not pb.is_valid:
        pb = pb.buffer(0)
    inter = pa.intersection(pb)
    if inter.is_empty:
        return None, 0.0
    if inter.geom_type == 'MultiPolygon':
        parts = []
        for g in inter.geoms:
            parts.append(np.array(g.exterior.coords))
        return parts, inter.area
    if inter.geom_type == 'Polygon':
        return [np.array(inter.exterior.coords)], inter.area
    return None, 0.0


# ────────────────────────────────────────────
# plotting
# ────────────────────────────────────────────

def draw_vehicle_box(ax):
    w, h = 0.60, 0.50
    rect = patches.Rectangle((-w/2, -h/2), w, h,
                              linewidth=2, edgecolor='gray',
                              facecolor='lightgray', alpha=0.25,
                              linestyle='--', label='Vehicle body')
    ax.add_patch(rect)


def plot_workspace_figure(
    left_ee_verts, right_ee_verts,
    left_all_verts, right_all_verts,
    T_left, T_right,
    title_prefix, out_prefix,
    left_ee_pts=None, right_ee_pts=None,
    left_all_pts=None, right_all_pts=None,
):
    for fig_idx, (l_verts, r_verts, tag, l_pts, r_pts) in enumerate([
        (left_ee_verts, right_ee_verts, "End-Effector", left_ee_pts, right_ee_pts),
        (left_all_verts, right_all_verts, "Full-Body (All Links)", left_all_pts, right_all_pts),
    ]):
        fig, axes = plt.subplots(1, 3, figsize=(21, 7))
        fig.suptitle(f"{title_prefix} — {tag} Reachable Workspace (XY projection, vehicle frame)",
                     fontsize=13, fontweight='bold')

        # Subplot 1: Left arm
        ax = axes[0]
        draw_vehicle_box(ax)
        if l_pts is not None and len(l_pts) > 0:
            ax.scatter(l_pts[:, 0], l_pts[:, 1], s=0.3, c='#AFC4E4', alpha=0.3, zorder=1)
        if len(l_verts) >= 3:
            poly = MplPolygon(l_verts, closed=True, facecolor='#AFC4E4', edgecolor='#3B6DAD',
                              alpha=0.4, linewidth=1.5, zorder=2, label='Left arm workspace')
            ax.add_patch(poly)
        ax.scatter(T_left[0, 3], T_left[1, 3], c='blue', marker='*', s=250,
                   zorder=5, label='Left arm base')
        ax.scatter(T_right[0, 3], T_right[1, 3], c='green', marker='*', s=250,
                   zorder=5, label='Right arm base')
        ax.set_title(f"Left Arm {tag} Workspace", fontsize=11)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=7)

        # Subplot 2: Right arm
        ax = axes[1]
        draw_vehicle_box(ax)
        if r_pts is not None and len(r_pts) > 0:
            ax.scatter(r_pts[:, 0], r_pts[:, 1], s=0.3, c='#BEE4C8', alpha=0.3, zorder=1)
        if len(r_verts) >= 3:
            poly = MplPolygon(r_verts, closed=True, facecolor='#BEE4C8', edgecolor='#2D8B4E',
                              alpha=0.4, linewidth=1.5, zorder=2, label='Right arm workspace')
            ax.add_patch(poly)
        ax.scatter(T_left[0, 3], T_left[1, 3], c='blue', marker='*', s=250,
                   zorder=5, label='Left arm base')
        ax.scatter(T_right[0, 3], T_right[1, 3], c='green', marker='*', s=250,
                   zorder=5, label='Right arm base')
        ax.set_title(f"Right Arm {tag} Workspace", fontsize=11)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=7)

        # Subplot 3: Both + Intersection
        ax = axes[2]
        draw_vehicle_box(ax)
        if len(l_verts) >= 3:
            poly_l = MplPolygon(l_verts, closed=True, facecolor='#AFC4E4', edgecolor='#3B6DAD',
                                alpha=0.3, linewidth=1.2, zorder=2, label='Left arm')
            ax.add_patch(poly_l)
        if len(r_verts) >= 3:
            poly_r = MplPolygon(r_verts, closed=True, facecolor='#BEE4C8', edgecolor='#2D8B4E',
                                alpha=0.3, linewidth=1.2, zorder=2, label='Right arm')
            ax.add_patch(poly_r)

        inter_parts, inter_area = compute_intersection_polygon(l_verts, r_verts)
        if inter_parts is not None:
            for ip in inter_parts:
                poly_inter = MplPolygon(ip, closed=True, facecolor='#FF6B6B',
                                        edgecolor='red', alpha=0.5, linewidth=2,
                                        zorder=3, label=f'Intersection (area={inter_area:.4f} m²)')
                ax.add_patch(poly_inter)
        ax.scatter(T_left[0, 3], T_left[1, 3], c='blue', marker='*', s=250,
                   zorder=5, label='Left arm base')
        ax.scatter(T_right[0, 3], T_right[1, 3], c='green', marker='*', s=250,
                   zorder=5, label='Right arm base')
        ax.set_title(f"Both Arms + Intersection ({tag})", fontsize=11)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=7)

        plt.tight_layout()

        suffix = "ee_workspace" if fig_idx == 0 else "fullbody_workspace"
        fpath = os.path.join(OUT_DIR, f"{out_prefix}_{suffix}.png")
        plt.savefig(fpath, dpi=200, bbox_inches='tight')
        print(f"Saved: {fpath}")
        plt.close(fig)


# ────────────────────────────────────────────
# comparison figure (old vs new side by side)
# ────────────────────────────────────────────

def plot_comparison(old_l_ee, old_r_ee, old_l_all, old_r_all,
                    new_l_ee, new_r_ee, new_l_all, new_r_all,
                    T_left, T_right):
    """
    绘制新旧构型对比图（2 行 × 3 列）：
    第一行：EE 可达空间（旧 / 新 / 叠加对比）
    第二行：全身可达空间（旧 / 新 / 叠加对比）
    """
    fig, axes = plt.subplots(2, 3, figsize=(21, 14))
    fig.suptitle("Old vs New URDF — Workspace Comparison (XY, vehicle frame)",
                 fontsize=14, fontweight='bold')

    row_data = [
        ("End-Effector", old_l_ee, old_r_ee, new_l_ee, new_r_ee),
        ("Full-Body", old_l_all, old_r_all, new_l_all, new_r_all),
    ]

    for row, (tag, ol, orr, nl, nr) in enumerate(row_data):
        # Col 0: old
        ax = axes[row][0]
        draw_vehicle_box(ax)
        if len(ol) >= 3:
            ax.add_patch(MplPolygon(ol, closed=True, fc='#AFC4E4', ec='#3B6DAD',
                                    alpha=0.4, lw=1.5, label='Old Left'))
        if len(orr) >= 3:
            ax.add_patch(MplPolygon(orr, closed=True, fc='#BEE4C8', ec='#2D8B4E',
                                    alpha=0.4, lw=1.5, label='Old Right'))
        inter_p, inter_a = compute_intersection_polygon(ol, orr)
        if inter_p:
            for ip in inter_p:
                ax.add_patch(MplPolygon(ip, closed=True, fc='#FF6B6B', ec='red',
                                        alpha=0.4, lw=1.5, label=f'Old Intersection ({inter_a:.4f} m²)'))
        ax.scatter(T_left[0,3], T_left[1,3], c='blue', marker='*', s=200, zorder=5)
        ax.scatter(T_right[0,3], T_right[1,3], c='green', marker='*', s=200, zorder=5)
        ax.set_title(f"Old URDF — {tag}", fontsize=11)
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys(), fontsize=7)

        # Col 1: new
        ax = axes[row][1]
        draw_vehicle_box(ax)
        if len(nl) >= 3:
            ax.add_patch(MplPolygon(nl, closed=True, fc='#AFC4E4', ec='#3B6DAD',
                                    alpha=0.4, lw=1.5, label='New Left'))
        if len(nr) >= 3:
            ax.add_patch(MplPolygon(nr, closed=True, fc='#BEE4C8', ec='#2D8B4E',
                                    alpha=0.4, lw=1.5, label='New Right'))
        inter_p2, inter_a2 = compute_intersection_polygon(nl, nr)
        if inter_p2:
            for ip in inter_p2:
                ax.add_patch(MplPolygon(ip, closed=True, fc='#FF6B6B', ec='red',
                                        alpha=0.4, lw=1.5, label=f'New Intersection ({inter_a2:.4f} m²)'))
        ax.scatter(T_left[0,3], T_left[1,3], c='blue', marker='*', s=200, zorder=5)
        ax.scatter(T_right[0,3], T_right[1,3], c='green', marker='*', s=200, zorder=5)
        ax.set_title(f"New URDF — {tag}", fontsize=11)
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys(), fontsize=7)

        # Col 2: overlay
        ax = axes[row][2]
        draw_vehicle_box(ax)
        if len(ol) >= 3:
            ax.add_patch(MplPolygon(ol, closed=True, fc='none', ec='#3B6DAD',
                                    alpha=0.6, lw=2, ls='--', label='Old Left'))
        if len(orr) >= 3:
            ax.add_patch(MplPolygon(orr, closed=True, fc='none', ec='#2D8B4E',
                                    alpha=0.6, lw=2, ls='--', label='Old Right'))
        if len(nl) >= 3:
            ax.add_patch(MplPolygon(nl, closed=True, fc='#AFC4E4', ec='#3B6DAD',
                                    alpha=0.25, lw=2, ls='-', label='New Left'))
        if len(nr) >= 3:
            ax.add_patch(MplPolygon(nr, closed=True, fc='#BEE4C8', ec='#2D8B4E',
                                    alpha=0.25, lw=2, ls='-', label='New Right'))
        ax.scatter(T_left[0,3], T_left[1,3], c='blue', marker='*', s=200, zorder=5)
        ax.scatter(T_right[0,3], T_right[1,3], c='green', marker='*', s=200, zorder=5)
        ax.set_title(f"Overlay (dashed=Old, solid=New) — {tag}", fontsize=11)
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys(), fontsize=7)

    for ax_row in axes:
        for ax in ax_row:
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")

    plt.tight_layout()
    fpath = os.path.join(OUT_DIR, "old_vs_new_comparison.png")
    plt.savefig(fpath, dpi=200, bbox_inches='tight')
    print(f"Saved: {fpath}")
    plt.close(fig)


# ────────────────────────────────────────────
# Load old URDF workspace (reuse FK from workspace_viz_old_urdf)
# ────────────────────────────────────────────

def _import_old_urdf_module():
    """Import workspace_viz_old_urdf as module from same directory."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "workspace_viz_old_urdf",
        os.path.join(_HERE, "workspace_viz_old_urdf.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ────────────────────────────────────────────
# main
# ────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Workspace Visualization — NEW URDF (berrybolter_arm)")
    print("=" * 60)

    # Build mounting transforms
    T_left = xyzrpy_to_T(LEFT_ARM_MOUNT_XYZ, LEFT_ARM_MOUNT_RPY)
    T_right = xyzrpy_to_T(RIGHT_ARM_MOUNT_XYZ, RIGHT_ARM_MOUNT_RPY)

    print(f"Left arm base (vehicle frame):  x={T_left[0,3]:.5f}, y={T_left[1,3]:.5f}")
    print(f"Right arm base (vehicle frame): x={T_right[0,3]:.5f}, y={T_right[1,3]:.5f}")

    # ── New URDF: parse single-arm chain ──
    # Chain: arm_base → base_link → link1 → ... → ee_link
    new_chain = parse_urdf_chain(NEW_URDF_PATH, "arm_base", "ee_link")
    print(f"\nNew URDF chain: {len(new_chain)} joints, "
          f"{sum(1 for j in new_chain if j.jtype != 'fixed')} actuated")
    for j in new_chain:
        if j.jtype != "fixed":
            print(f"  {j.name}: {j.jtype}, limits=[{j.lower:.4f}, {j.upper:.4f}]")

    print("\nSampling workspace for new arm (FK)...")
    n_samples = 100000
    new_ee_local, new_all_local, new_link_dict = compute_workspace(new_chain, n_samples)
    print(f"  EE points: {len(new_ee_local)}, All link points: {len(new_all_local)}")

    # Transform to vehicle frame
    new_left_ee = transform_points_2d(new_ee_local, T_left)
    new_left_all = transform_points_2d(new_all_local, T_left)
    new_right_ee = transform_points_2d(new_ee_local, T_right)
    new_right_all = transform_points_2d(new_all_local, T_right)

    print(f"New Left EE range: x=[{new_left_ee[:,0].min():.3f}, {new_left_ee[:,0].max():.3f}], "
          f"y=[{new_left_ee[:,1].min():.3f}, {new_left_ee[:,1].max():.3f}]")
    print(f"New Right EE range: x=[{new_right_ee[:,0].min():.3f}, {new_right_ee[:,0].max():.3f}], "
          f"y=[{new_right_ee[:,1].min():.3f}, {new_right_ee[:,1].max():.3f}]")

    # ── Compute boundaries ──
    print("\nComputing convex hull boundaries (new URDF)...")
    new_l_ee_hull = compute_alpha_shape(new_left_ee, 0)
    new_r_ee_hull = compute_alpha_shape(new_right_ee, 0)
    new_l_all_hull = compute_alpha_shape(new_left_all, 0)
    new_r_all_hull = compute_alpha_shape(new_right_all, 0)

    # ── Plot new URDF workspace ──
    print("\nGenerating new URDF plots (convex hull)...")
    plot_workspace_figure(
        new_l_ee_hull, new_r_ee_hull,
        new_l_all_hull, new_r_all_hull,
        T_left, T_right,
        title_prefix="New URDF (berrybolter_arm)",
        out_prefix="new_urdf",
        left_ee_pts=new_left_ee, right_ee_pts=new_right_ee,
        left_all_pts=new_left_all, right_all_pts=new_right_all,
    )

    # ── Concave alpha-shape boundaries ──
    print("Computing concave alpha-shape boundaries (new URDF)...")
    alpha_ee, alpha_all = 8.0, 5.0
    new_l_ee_alpha = compute_alpha_shape(new_left_ee, alpha_ee)
    new_r_ee_alpha = compute_alpha_shape(new_right_ee, alpha_ee)
    new_l_all_alpha = compute_alpha_shape(new_left_all, alpha_all)
    new_r_all_alpha = compute_alpha_shape(new_right_all, alpha_all)

    print("Generating new URDF plots (alpha-shape)...")
    plot_workspace_figure(
        new_l_ee_alpha, new_r_ee_alpha,
        new_l_all_alpha, new_r_all_alpha,
        T_left, T_right,
        title_prefix="New URDF (concave alpha-shape)",
        out_prefix="new_urdf_concave",
        left_ee_pts=new_left_ee, right_ee_pts=new_right_ee,
        left_all_pts=new_left_all, right_all_pts=new_right_all,
    )

    # ── Old URDF workspace (for comparison) ──
    print("\n--- Also computing OLD URDF workspace for comparison ---")
    old_mod = _import_old_urdf_module()

    old_tree = ET.parse(OLD_URDF_PATH)
    old_root = old_tree.getroot()

    def get_old_joint_T(jname):
        j = old_root.find(f"./joint[@name='{jname}']")
        o = j.find("origin")
        xyz = [float(v) for v in o.attrib.get("xyz", "0 0 0").split()]
        rpy = [float(v) for v in o.attrib.get("rpy", "0 0 0").split()]
        return xyzrpy_to_T(xyz, rpy)

    T_left_old = get_old_joint_T("vehicle_to_left_arm")
    T_right_old = get_old_joint_T("vehicle_to_right_arm")

    old_chain = old_mod.parse_urdf_chain(OLD_URDF_PATH, "left_arm_base_link", "left_arm_ee_link")
    old_ee_local, old_all_local, _ = old_mod.compute_workspace(old_chain, n_samples)

    old_left_ee = old_mod.transform_points_2d(old_ee_local, T_left_old)
    old_left_all = old_mod.transform_points_2d(old_all_local, T_left_old)
    old_right_ee = old_mod.transform_points_2d(old_ee_local, T_right_old)
    old_right_all = old_mod.transform_points_2d(old_all_local, T_right_old)

    old_l_ee_hull = compute_alpha_shape(old_left_ee, 0)
    old_r_ee_hull = compute_alpha_shape(old_right_ee, 0)
    old_l_all_hull = compute_alpha_shape(old_left_all, 0)
    old_r_all_hull = compute_alpha_shape(old_right_all, 0)

    # ── Comparison plot ──
    print("\nGenerating OLD vs NEW comparison plot...")
    plot_comparison(
        old_l_ee_hull, old_r_ee_hull, old_l_all_hull, old_r_all_hull,
        new_l_ee_hull, new_r_ee_hull, new_l_all_hull, new_r_all_hull,
        T_left, T_right,
    )

    print("\nDone! All plots saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
