#!/usr/bin/env python3
"""
workspace_viz_new_urdf.py
=========================
基于 **新构型** ``urdf/berrybolter_arm.urdf`` 的 FK 可达空间可视化。

该 URDF 是单臂描述（含 meshes），左右臂结构完全相同，
双臂安装位姿沿用 ``dual_arm_ik_xy_centered.urdf`` 中的
``vehicle_to_left_arm`` / ``vehicle_to_right_arm`` 固定关节。

脚本只对 **单臂** 做 FK 采样，另一臂结果通过安装位姿平移/旋转得到。

输出图表：
1. EE 可达空间（凸包 + alpha-shape 版本）
2. 全身（All Links）可达空间
3. **每个关节/连杆的可达区域**——用不同颜色区分各关节的覆盖范围
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
NEW_URDF_PATH = os.path.join(_HERE, "..", "urdf", "berrybolter_arm.urdf")
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
    """
    Return a Shapely geometry (Polygon/MultiPolygon) that may contain
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
    Convert a Shapely Polygon (possibly with holes) into a matplotlib Path.
    """
    from matplotlib.path import Path as MplPath
    from shapely.geometry import Polygon as ShapelyPolygon, MultiPolygon

    codes_all = []
    verts_all = []

    def _ring_to_codes_verts(ring_coords):
        coords = list(ring_coords)
        n = len(coords)
        codes = [MplPath.MOVETO] + [MplPath.LINETO] * (n - 2) + [MplPath.CLOSEPOLY]
        return coords, codes

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
# Per-joint colors — high-contrast, saturated, carefully ordered
# ────────────────────────────────────────────
JOINT_COLORS = [
    '#888888',  # base_link  (fixed) — gray
    '#FF6600',  # link1 — orange       (joint1, prismatic)
    '#00CC00',  # link2 — bright green  (joint2, revolute)
    '#FF0000',  # link3 — red           (joint3, prismatic)
    '#AA00FF',  # link4 — violet        (joint4, revolute)
    '#00AAFF',  # link5 — sky blue      (joint5, revolute)
    '#FF00AA',  # link6 — magenta       (joint6, revolute)
    '#CCCC00',  # link7 — dark yellow   (joint7, revolute)
    '#0000FF',  # ee_link (fixed) — blue
]


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
    for fig_idx, (l_shape, r_shape, tag, l_pts, r_pts) in enumerate([
        (left_ee_shape, right_ee_shape, "End-Effector", left_ee_pts, right_ee_pts),
        (left_all_shape, right_all_shape, "Full-Body (All Links)", left_all_pts, right_all_pts),
    ]):
        fig, axes = plt.subplots(1, 3, figsize=(21, 7))
        fig.suptitle(f"{title_prefix} — {tag} Reachable Workspace (XY projection, vehicle frame)",
                     fontsize=13, fontweight='bold')

        # Subplot 1: Left arm
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

        # Subplot 2: Right arm
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

        # Subplot 3: Both + Intersection
        ax = axes[2]
        draw_vehicle_box(ax)
        _draw_workspace_on_ax(ax, l_shape, l_pts, '#AFC4E4', '#3B6DAD',
                              'Left arm', outline_only=True)
        _draw_workspace_on_ax(ax, r_shape, r_pts, '#BEE4C8', '#2D8B4E',
                              'Right arm', outline_only=True)

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


def plot_per_joint_workspace(link_names, link_pts_left, link_pts_right,
                             T_left, T_right, chain, out_prefix):
    """
    为每个关节/连杆的可达区域用不同颜色画在同一张图上，
    生成 2 个子图（左臂 / 右臂），可以直观看出各关节构型的覆盖范围。

    绘制顺序：EE（最外层）先画 → 基座（最内层）后画，
    这样内部关节的点不会被外部关节的大量散点覆盖。
    """
    # Build label map: link name → joint name + type
    joint_label = {}
    for j in chain:
        jt = j.jtype[0].upper()  # R/P/F
        joint_label[j.child] = f"{j.child} ({j.name}, {jt})"

    n_links = len(link_names)

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    fig.suptitle("Per-Joint Reachable Area — berrybolter_arm (XY, vehicle frame)",
                 fontsize=14, fontweight='bold')

    for ax_idx, (arm_tag, pts_dict, T_mount) in enumerate([
        ("Left Arm", link_pts_left, T_left),
        ("Right Arm", link_pts_right, T_right),
    ]):
        ax = axes[ax_idx]
        draw_vehicle_box(ax)

        # Draw in REVERSE order: EE first (bottom), base_link last (top)
        # so that inner joints (smaller coverage) are visible on top
        for draw_i, i in enumerate(reversed(range(n_links))):
            lname = link_names[i]
            pts = pts_dict.get(lname)
            if pts is None or len(pts) == 0:
                continue
            color = JOINT_COLORS[i % len(JOINT_COLORS)]
            label = joint_label.get(lname, lname)
            # Inner joints get slightly larger points for visibility
            sz = 0.6 + draw_i * 0.15
            ax.scatter(pts[:, 0], pts[:, 1], s=sz, c=color, alpha=0.5,
                       zorder=1 + draw_i, rasterized=True, label=label)

        # Arm base markers
        ax.scatter(T_left[0, 3], T_left[1, 3], c='blue', marker='*', s=300,
                   zorder=20, label='Left arm base', edgecolors='black', linewidths=0.5)
        ax.scatter(T_right[0, 3], T_right[1, 3], c='green', marker='*', s=300,
                   zorder=20, label='Right arm base', edgecolors='black', linewidths=0.5)

        ax.set_title(f"{arm_tag} — Per-Joint Workspace", fontsize=12)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

        # Build legend in chain order (base → EE), not draw order
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        # Reorder: chain order first, then base markers
        ordered_handles, ordered_labels = [], []
        for i, lname in enumerate(link_names):
            lab = joint_label.get(lname, lname)
            if lab in by_label:
                ordered_labels.append(lab)
                ordered_handles.append(by_label.pop(lab))
        # Append remaining (base markers, vehicle box)
        for lab, h in by_label.items():
            ordered_labels.append(lab)
            ordered_handles.append(h)
        ax.legend(ordered_handles, ordered_labels,
                  loc='upper left', fontsize=6.5, markerscale=5,
                  ncol=1, framealpha=0.85)

    plt.tight_layout()
    fpath = os.path.join(OUT_DIR, f"{out_prefix}_per_joint_workspace.png")
    plt.savefig(fpath, dpi=200, bbox_inches='tight')
    print(f"Saved: {fpath}")
    plt.close(fig)


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

    # Ordered link names (chain order, base → EE)
    link_names = [j.child for j in new_chain]
    print(f"  Links (chain order): {link_names}")

    # Transform to vehicle frame
    new_left_ee = transform_points_2d(new_ee_local, T_left)
    new_left_all = transform_points_2d(new_all_local, T_left)
    new_right_ee = transform_points_2d(new_ee_local, T_right)
    new_right_all = transform_points_2d(new_all_local, T_right)

    # Per-link points in vehicle frame
    link_pts_left = {}
    link_pts_right = {}
    for lname, pts_local in new_link_dict.items():
        link_pts_left[lname] = transform_points_2d(pts_local, T_left)
        link_pts_right[lname] = transform_points_2d(pts_local, T_right)

    print(f"New Left EE range: x=[{new_left_ee[:,0].min():.3f}, {new_left_ee[:,0].max():.3f}], "
          f"y=[{new_left_ee[:,1].min():.3f}, {new_left_ee[:,1].max():.3f}]")
    print(f"New Right EE range: x=[{new_right_ee[:,0].min():.3f}, {new_right_ee[:,0].max():.3f}], "
          f"y=[{new_right_ee[:,1].min():.3f}, {new_right_ee[:,1].max():.3f}]")

    # ── Compute boundaries ──
    print("\nComputing convex hull boundaries...")
    new_l_ee_hull = compute_alpha_shape(new_left_ee, 0)
    new_r_ee_hull = compute_alpha_shape(new_right_ee, 0)
    new_l_all_hull = compute_alpha_shape(new_left_all, 0)
    new_r_all_hull = compute_alpha_shape(new_right_all, 0)

    # ── Plot workspace (convex hull) ──
    print("\nGenerating workspace plots (convex hull)...")
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
    print("Computing concave alpha-shape boundaries...")
    alpha_ee, alpha_all = 8.0, 5.0
    new_l_ee_alpha = compute_alpha_shape(new_left_ee, alpha_ee)
    new_r_ee_alpha = compute_alpha_shape(new_right_ee, alpha_ee)
    new_l_all_alpha = compute_alpha_shape(new_left_all, alpha_all)
    new_r_all_alpha = compute_alpha_shape(new_right_all, alpha_all)

    print("Generating workspace plots (alpha-shape)...")
    plot_workspace_figure(
        new_l_ee_alpha, new_r_ee_alpha,
        new_l_all_alpha, new_r_all_alpha,
        T_left, T_right,
        title_prefix="New URDF (concave alpha-shape)",
        out_prefix="new_urdf_concave",
        left_ee_pts=new_left_ee, right_ee_pts=new_right_ee,
        left_all_pts=new_left_all, right_all_pts=new_right_all,
    )

    # ── Per-joint workspace visualization ──
    print("\nGenerating per-joint workspace plot...")
    plot_per_joint_workspace(
        link_names, link_pts_left, link_pts_right,
        T_left, T_right, new_chain, out_prefix="new_urdf",
    )

    print("\nDone! All plots saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
