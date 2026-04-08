#!/usr/bin/env python3
"""
workspace_viz_old_urdf.py
=========================
基于 **旧构型** ``urdf/dual_arm_ik_xy_centered.urdf`` 的 FK 可达空间可视化。

该 URDF 为简化版（无 meshes），左右臂结构完全对称，仅安装位姿不同。
脚本只对 **左臂** 做 FK 采样，右臂的结果通过安装位姿平移/旋转得到。

输出两张图：
1. 末端执行器（EE）二维可达范围 + 双臂交集
2. 全部连杆/关节的二维扫掠范围 + 双臂交集
"""

import os
import sys
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial import ConvexHull

# ────────────────────────────────────────────
# paths
# ────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
URDF_PATH = os.path.join(_HERE, "..", "urdf", "dual_arm_ik_xy_centered.urdf")
OUT_DIR = os.path.join(_HERE, "results")
os.makedirs(OUT_DIR, exist_ok=True)

# ────────────────────────────────────────────
# basic transforms
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
# URDF chain parser (simplified, old URDF)
# ────────────────────────────────────────────

class JointInfo:
    """One joint parsed from URDF."""
    def __init__(self, name, jtype, origin_T, axis, lower, upper, parent, child):
        self.name = name
        self.jtype = jtype          # "revolute" / "prismatic" / "fixed"
        self.origin_T = origin_T    # 4x4 static transform
        self.axis = axis            # (3,) unit vector
        self.lower = lower
        self.upper = upper
        self.parent = parent
        self.child = child


def parse_urdf_chain(urdf_path, base_link, tip_link, prefix=""):
    """
    解析 URDF 中从 base_link 到 tip_link 的运动链，返回 JointInfo 列表。
    prefix: 用于匹配 link/joint 名字的前缀（如 'left_arm_'）
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # Build parent->child map via joints
    joints_by_child = {}
    for j in root.findall("joint"):
        child = j.find("child").attrib["link"]
        joints_by_child[child] = j

    # Walk from tip to base to find chain
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
    """Rodrigues rotation matrix for arbitrary unit axis."""
    axis = axis / (np.linalg.norm(axis) + 1e-15)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def fk_all_link_origins(chain, q_vec):
    """
    给定关节角向量 q_vec，返回 chain 中每个 link origin 在 base_link 坐标系中的位置。
    返回: list of (link_name, (4,4) transform)
    """
    T = np.eye(4)
    results = []
    qi = 0  # index into q_vec (only for actuated joints)
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
        # fixed: no variable
        results.append((joint.child, T.copy()))
    return results


def get_actuated_limits(chain):
    """Return (lower, upper) arrays for actuated joints only."""
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
    """
    采样关节空间。
    mode='random': 在关节限位内随机采样
    mode='boundary': 额外加上各关节取极值的组合（用于勾勒外轮廓）
    """
    lower, upper = get_actuated_limits(chain)
    ndof = len(lower)

    configs = []
    if mode in ("random", "both"):
        rng = np.random.default_rng(42)
        for _ in range(n_samples):
            q = lower + (upper - lower) * rng.random(ndof)
            configs.append(q)

    if mode in ("boundary", "both"):
        # Enumerate corners: each joint at min or max
        # For 7 DOF that's 128 — manageable
        from itertools import product
        for combo in product(*[(lo, hi) for lo, hi in zip(lower, upper)]):
            configs.append(np.array(combo))

        # Also add mid-points with each joint at extreme
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
    """
    返回:
      ee_points_xy: (N, 2) 末端执行器在 base_link 系的 (x, y)
      all_points_xy: (M, 2) 所有连杆 origin 的 (x, y)（含 EE）
      link_points_dict: {link_name: (K, 2)} 各连杆的 (x, y) 点集
    """
    configs = sample_joint_configs(chain, n_samples, mode="both")

    ee_points = []
    all_points = []
    link_points_dict = {}

    for q in configs:
        frames = fk_all_link_origins(chain, q)
        for lname, T in frames:
            p = T[:3, 3]
            xy = p[:2]
            all_points.append(xy)
            if lname not in link_points_dict:
                link_points_dict[lname] = []
            link_points_dict[lname].append(xy)

        # EE is the last frame
        ee_T = frames[-1][1]
        ee_points.append(ee_T[:2, 3])

    ee_points = np.array(ee_points)
    all_points = np.array(all_points)
    for k in link_points_dict:
        link_points_dict[k] = np.array(link_points_dict[k])

    return ee_points, all_points, link_points_dict


# ────────────────────────────────────────────
# transform points from base_link frame → vehicle frame
# ────────────────────────────────────────────

def transform_points_2d(points_xy, T_vehicle_from_arm):
    """
    将 (N,2) 点集从 arm-base frame 变换到 vehicle frame 的 (x,y)。
    T_vehicle_from_arm: 4x4 安装变换
    """
    N = len(points_xy)
    ones = np.ones((N, 1))
    zeros = np.zeros((N, 1))
    pts_h = np.hstack([points_xy, zeros, ones])  # (N, 4)
    pts_vehicle = (T_vehicle_from_arm @ pts_h.T).T  # (N, 4)
    return pts_vehicle[:, :2]


# ────────────────────────────────────────────
# convex hull helper
# ────────────────────────────────────────────

def safe_convex_hull(pts):
    """Return ConvexHull or None if degenerate."""
    if len(pts) < 3:
        return None
    try:
        return ConvexHull(pts)
    except Exception:
        return None


def hull_polygon_xy(hull):
    """Return ordered boundary vertices from a ConvexHull."""
    if hull is None:
        return np.empty((0, 2))
    verts = hull.points[hull.vertices]
    return verts


# ────────────────────────────────────────────
# concave hull (alpha shape) helper
# ────────────────────────────────────────────

def _downsample_for_alpha(pts, max_pts=5000):
    """Downsample point cloud to max_pts for alpha shape (via random selection)."""
    if len(pts) <= max_pts:
        return pts
    rng = np.random.default_rng(0)
    idx = rng.choice(len(pts), max_pts, replace=False)
    return pts[idx]


def compute_alpha_shape(pts, alpha=0.0):
    """
    Use alphashape or fallback to convex hull.
    alpha=0 → convex hull; larger alpha → tighter boundary.
    Points are downsampled to keep computation fast.

    Returns a Shapely geometry (Polygon/MultiPolygon) that may contain
    interior holes, or None if degenerate.
    """
    from shapely.geometry import Polygon as ShapelyPolygon, MultiPolygon
    if alpha == 0:
        hull = safe_convex_hull(pts)
        if hull is None:
            return None
        verts = hull_polygon_xy(hull)
        if len(verts) < 3:
            return None
        return ShapelyPolygon(verts)
    try:
        import alphashape as ash
        pts_ds = _downsample_for_alpha(pts, max_pts=3000)
        shape = ash.alphashape(pts_ds, alpha)
        if shape is None or shape.is_empty:
            hull = safe_convex_hull(pts)
            if hull is None:
                return None
            return ShapelyPolygon(hull_polygon_xy(hull))
        if isinstance(shape, MultiPolygon):
            return max(shape.geoms, key=lambda g: g.area)
        if isinstance(shape, ShapelyPolygon):
            return shape
        # Fallback
        hull = safe_convex_hull(pts)
        if hull is None:
            return None
        return ShapelyPolygon(hull_polygon_xy(hull))
    except Exception:
        hull = safe_convex_hull(pts)
        if hull is None:
            return None
        return ShapelyPolygon(hull_polygon_xy(hull))


def compute_intersection_polygon(shape_a, shape_b):
    """
    Compute intersection of two Shapely geometries.
    Returns (intersection_shapely_geom_or_None, area).
    """
    if shape_a is None or shape_b is None:
        return None, 0.0
    if not shape_a.is_valid:
        shape_a = shape_a.buffer(0)
    if not shape_b.is_valid:
        shape_b = shape_b.buffer(0)
    inter = shape_a.intersection(shape_b)
    if inter.is_empty:
        return None, 0.0
    return inter, inter.area


def _shapely_to_mpl_path(geom):
    """
    Convert a Shapely Polygon (possibly with holes) into a matplotlib Path
    so that interior holes are rendered correctly.
    """
    from matplotlib.path import Path as MplPath
    import matplotlib.patches as mpatches

    codes_all = []
    verts_all = []

    def _ring_to_codes_verts(ring_coords):
        coords = list(ring_coords)
        n = len(coords)
        codes = [MplPath.MOVETO] + [MplPath.LINETO] * (n - 2) + [MplPath.CLOSEPOLY]
        return coords, codes

    from shapely.geometry import Polygon as ShapelyPolygon, MultiPolygon
    polygons = []
    if isinstance(geom, MultiPolygon):
        polygons = list(geom.geoms)
    elif isinstance(geom, ShapelyPolygon):
        polygons = [geom]
    else:
        return None

    for poly in polygons:
        ext_coords, ext_codes = _ring_to_codes_verts(poly.exterior.coords)
        verts_all.extend(ext_coords)
        codes_all.extend(ext_codes)
        for interior in poly.interiors:
            int_coords, int_codes = _ring_to_codes_verts(interior.coords)
            verts_all.extend(int_coords)
            codes_all.extend(int_codes)

    if not verts_all:
        return None
    return MplPath(verts_all, codes_all)


def shapely_to_mpl_patch(geom, **kwargs):
    """Create a matplotlib PathPatch from a Shapely geometry (with holes)."""
    from matplotlib.patches import PathPatch
    path = _shapely_to_mpl_path(geom)
    if path is None:
        return None
    return PathPatch(path, **kwargs)


# ────────────────────────────────────────────
# plotting
# ────────────────────────────────────────────

def draw_vehicle_box(ax):
    """Draw simplified vehicle body (from URDF visual: box 0.60 x 0.50)."""
    w, h = 0.60, 0.50
    rect = patches.Rectangle((-w/2, -h/2), w, h,
                              linewidth=2, edgecolor='gray',
                              facecolor='lightgray', alpha=0.25,
                              linestyle='--', label='Vehicle body')
    ax.add_patch(rect)


def _draw_workspace_on_ax(ax, shape, pts, facecolor, edgecolor, label,
                          alpha_fill=0.35, outline_only=False):
    """
    Draw a workspace region on an axes.

    When outline_only=True (used for single-arm subplots):
      - Scatter the raw FK points (density naturally reveals inner holes)
      - Draw boundary as outline only (no fill), so the scatter hole is visible

    When outline_only=False (used for combined/intersection subplots):
      - Draw filled boundary polygon (with interior holes rendered via PathPatch)
    """
    if pts is not None and len(pts) > 0:
        ax.scatter(pts[:, 0], pts[:, 1], s=0.3, c=facecolor, alpha=0.3,
                   zorder=1, rasterized=True)
    if shape is not None:
        fc = 'none' if outline_only else facecolor
        af = 1.0 if outline_only else alpha_fill
        lw = 2.0 if outline_only else 1.5
        patch = shapely_to_mpl_patch(
            shape, facecolor=fc, edgecolor=edgecolor,
            alpha=af, linewidth=lw, zorder=2, label=label)
        if patch is not None:
            ax.add_patch(patch)


def plot_workspace_figure(
    left_ee_shape, right_ee_shape,
    left_all_shape, right_all_shape,
    T_left, T_right,
    title_prefix, out_prefix,
    left_ee_pts=None, right_ee_pts=None,
    left_all_pts=None, right_all_pts=None,
):
    """
    绘制两张图:
    Figure 1: EE workspace (with holes for unreachable inner region)
    Figure 2: Full body workspace
    """
    for fig_idx, (l_shape, r_shape, tag, l_pts, r_pts) in enumerate([
        (left_ee_shape, right_ee_shape, "End-Effector", left_ee_pts, right_ee_pts),
        (left_all_shape, right_all_shape, "Full-Body (All Links)", left_all_pts, right_all_pts),
    ]):
        fig, axes = plt.subplots(1, 3, figsize=(21, 7))
        fig.suptitle(f"{title_prefix} — {tag} Reachable Workspace (XY projection, vehicle frame)",
                     fontsize=13, fontweight='bold')

        # --- Subplot 1: Left arm ---
        ax = axes[0]
        draw_vehicle_box(ax)
        _draw_workspace_on_ax(ax, l_shape, l_pts, '#AFC4E4', '#3B6DAD',
                              'Left arm workspace', outline_only=True)
        ax.scatter(T_left[0, 3], T_left[1, 3], c='blue', marker='*', s=250,
                   zorder=5, label='Left arm base')
        ax.scatter(T_right[0, 3], T_right[1, 3], c='green', marker='*', s=250,
                   zorder=5, label='Right arm base')
        ax.set_title(f"Left Arm {tag} Workspace", fontsize=11)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=7)

        # --- Subplot 2: Right arm ---
        ax = axes[1]
        draw_vehicle_box(ax)
        _draw_workspace_on_ax(ax, r_shape, r_pts, '#BEE4C8', '#2D8B4E',
                              'Right arm workspace', outline_only=True)
        ax.scatter(T_left[0, 3], T_left[1, 3], c='blue', marker='*', s=250,
                   zorder=5, label='Left arm base')
        ax.scatter(T_right[0, 3], T_right[1, 3], c='green', marker='*', s=250,
                   zorder=5, label='Right arm base')
        ax.set_title(f"Right Arm {tag} Workspace", fontsize=11)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=7)

        # --- Subplot 3: Both + Intersection ---
        ax = axes[2]
        draw_vehicle_box(ax)
        # Show scatter from both arms so holes are visible
        _draw_workspace_on_ax(ax, l_shape, l_pts, '#AFC4E4', '#3B6DAD',
                              'Left arm', outline_only=True)
        _draw_workspace_on_ax(ax, r_shape, r_pts, '#BEE4C8', '#2D8B4E',
                              'Right arm', outline_only=True)

        # Intersection
        inter_geom, inter_area = compute_intersection_polygon(l_shape, r_shape)
        if inter_geom is not None:
            inter_patch = shapely_to_mpl_patch(
                inter_geom, facecolor='#FF6B6B', edgecolor='red',
                alpha=0.5, linewidth=2, zorder=3,
                label=f'Intersection (area={inter_area:.4f} m²)')
            if inter_patch is not None:
                ax.add_patch(inter_patch)

        ax.scatter(T_left[0, 3], T_left[1, 3], c='blue', marker='*', s=250,
                   zorder=5, label='Left arm base')
        ax.scatter(T_right[0, 3], T_right[1, 3], c='green', marker='*', s=250,
                   zorder=5, label='Right arm base')
        ax.set_title(f"Both Arms + Intersection ({tag})", fontsize=11)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
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
# main
# ────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Workspace Visualization — OLD URDF (dual_arm_ik_xy_centered)")
    print("=" * 60)

    # Parse base transforms
    tree = ET.parse(URDF_PATH)
    root = tree.getroot()

    def get_joint_T(joint_name):
        j = root.find(f"./joint[@name='{joint_name}']")
        origin = j.find("origin")
        xyz = [float(v) for v in origin.attrib.get("xyz", "0 0 0").split()]
        rpy = [float(v) for v in origin.attrib.get("rpy", "0 0 0").split()]
        return xyzrpy_to_T(xyz, rpy)

    T_left = get_joint_T("vehicle_to_left_arm")
    T_right = get_joint_T("vehicle_to_right_arm")

    print(f"Left arm base (vehicle frame): x={T_left[0,3]:.5f}, y={T_left[1,3]:.5f}, z={T_left[2,3]:.5f}")
    print(f"Right arm base (vehicle frame): x={T_right[0,3]:.5f}, y={T_right[1,3]:.5f}, z={T_right[2,3]:.5f}")

    # Parse the left arm chain (base_link → ee_link)
    left_chain = parse_urdf_chain(URDF_PATH, "left_arm_base_link", "left_arm_ee_link")
    print(f"Left arm chain: {len(left_chain)} joints, "
          f"{sum(1 for j in left_chain if j.jtype != 'fixed')} actuated")
    for j in left_chain:
        if j.jtype != "fixed":
            print(f"  {j.name}: {j.jtype}, limits=[{j.lower:.4f}, {j.upper:.4f}]")

    # Compute workspace for left arm (in left_arm_base_link frame)
    print("\nSampling workspace for left arm (FK)...")
    n_samples = 100000
    ee_pts_local, all_pts_local, link_pts_dict = compute_workspace(left_chain, n_samples)
    print(f"  EE points: {len(ee_pts_local)}, All link points: {len(all_pts_local)}")

    # Transform to vehicle frame
    left_ee_vehicle = transform_points_2d(ee_pts_local, T_left)
    left_all_vehicle = transform_points_2d(all_pts_local, T_left)

    # For right arm: same structure, different mounting.
    # We compute FK in the same local frame, then transform with T_right.
    right_ee_vehicle = transform_points_2d(ee_pts_local, T_right)
    right_all_vehicle = transform_points_2d(all_pts_local, T_right)

    print(f"Left EE range: x=[{left_ee_vehicle[:,0].min():.3f}, {left_ee_vehicle[:,0].max():.3f}], "
          f"y=[{left_ee_vehicle[:,1].min():.3f}, {left_ee_vehicle[:,1].max():.3f}]")
    print(f"Right EE range: x=[{right_ee_vehicle[:,0].min():.3f}, {right_ee_vehicle[:,0].max():.3f}], "
          f"y=[{right_ee_vehicle[:,1].min():.3f}, {right_ee_vehicle[:,1].max():.3f}]")

    # Compute alpha shapes for boundaries
    print("\nComputing boundary shapes...")
    # Use alpha=0 for convex hull (simpler, more robust)
    alpha_ee = 0  # convex hull
    alpha_all = 0

    left_ee_boundary = compute_alpha_shape(left_ee_vehicle, alpha_ee)
    right_ee_boundary = compute_alpha_shape(right_ee_vehicle, alpha_ee)
    left_all_boundary = compute_alpha_shape(left_all_vehicle, alpha_all)
    right_all_boundary = compute_alpha_shape(right_all_vehicle, alpha_all)

    def _shape_info(s):
        if s is None:
            return "None"
        n_ext = len(s.exterior.coords) if hasattr(s, 'exterior') else 0
        n_holes = len(list(s.interiors)) if hasattr(s, 'interiors') else 0
        return f"{n_ext} ext verts, {n_holes} holes, area={s.area:.4f} m²"

    print(f"  Left EE boundary: {_shape_info(left_ee_boundary)}")
    print(f"  Right EE boundary: {_shape_info(right_ee_boundary)}")
    print(f"  Left full-body boundary: {_shape_info(left_all_boundary)}")
    print(f"  Right full-body boundary: {_shape_info(right_all_boundary)}")

    # Plot
    print("\nGenerating plots...")
    plot_workspace_figure(
        left_ee_boundary, right_ee_boundary,
        left_all_boundary, right_all_boundary,
        T_left, T_right,
        title_prefix="Old URDF (dual_arm_ik_xy_centered)",
        out_prefix="old_urdf",
        left_ee_pts=left_ee_vehicle,
        right_ee_pts=right_ee_vehicle,
        left_all_pts=left_all_vehicle,
        right_all_pts=right_all_vehicle,
    )

    # Also try alpha shapes with higher alpha for tighter boundary
    print("\nComputing tighter (concave) boundary shapes with alpha=8...")
    alpha_ee_tight = 8.0
    alpha_all_tight = 5.0

    left_ee_boundary_t = compute_alpha_shape(left_ee_vehicle, alpha_ee_tight)
    right_ee_boundary_t = compute_alpha_shape(right_ee_vehicle, alpha_ee_tight)
    left_all_boundary_t = compute_alpha_shape(left_all_vehicle, alpha_all_tight)
    right_all_boundary_t = compute_alpha_shape(right_all_vehicle, alpha_all_tight)

    plot_workspace_figure(
        left_ee_boundary_t, right_ee_boundary_t,
        left_all_boundary_t, right_all_boundary_t,
        T_left, T_right,
        title_prefix="Old URDF (concave alpha-shape)",
        out_prefix="old_urdf_concave",
        left_ee_pts=left_ee_vehicle,
        right_ee_pts=right_ee_vehicle,
        left_all_pts=left_all_vehicle,
        right_all_pts=right_all_vehicle,
    )

    print("\nDone! All plots saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
