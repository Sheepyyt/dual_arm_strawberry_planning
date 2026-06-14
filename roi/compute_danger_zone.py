#!/usr/bin/env python3
"""
compute_danger_zone.py — 基于全身扫掠包络的危险区求解与可视化
================================================================
使用 dual_arm_ik_xy_centered.urdf 的运动学链，离线采样大量关节构型，
对每条连杆做"有厚度线段"（胶囊体）近似，投影到 XY 平面后生成：
  - E_L : 左臂全关节空间扫掠包络 (2D occupancy)
  - E_R : 右臂全关节空间扫掠包络
  - E_I = E_L ∩ E_R : 双臂重叠危险区

再分别与 ROI 表上半区 (x ∈ [-0.5,0.5], y ∈ [0.25,0.65])
和下半区 (x ∈ [-0.5,0.5], y ∈ [-0.65,-0.25]) 做交集，
得到后续规划需要关注的干涉目标区域。
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# 共享几何常数统一来自仓库根目录的 config.py（单一事实来源）
from config import (
    URDF_RELPATH,
    GRID_RES,        # Occupancy grid resolution (metres per cell) = 0.005
    LINK_RADIUS,     # Capsule radius (m), 把每节连杆近似成有厚度的线段 = 0.03
    ROI_UPPER,       # ROI 上半区
    ROI_LOWER,       # ROI 下半区
)

URDF_PATH = os.path.join(PROJECT_ROOT, URDF_RELPATH)
RESULT_DIR = os.path.join(SCRIPT_DIR, "results", "danger")

# Grid bounding box (vehicle frame, XY plane) — generous margins（危险区专用，非共享）
GRID_X_MIN, GRID_X_MAX = -0.80, 0.80
GRID_Y_MIN, GRID_Y_MAX = -0.90, 0.90

# Number of random joint configurations to sample per arm
N_SAMPLES = 200_000


# ===========================================================================
# Rotation helpers
# ===========================================================================
def _rpy_to_rot(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def _xyzrpy_to_T(xyz, rpy):
    T = np.eye(4)
    T[:3, :3] = _rpy_to_rot(*rpy)
    T[:3, 3] = xyz
    return T


# ===========================================================================
# URDF chain parser
# ===========================================================================
class URDFArmChain:
    """
    Parse one arm's kinematic chain from the URDF and provide vectorised FK
    that returns all joint-frame origins (in vehicle frame) for a batch of
    joint configurations.
    """

    def __init__(self, urdf_path, base_joint_name, joint_prefix):
        tree = ET.parse(urdf_path)
        root = tree.getroot()

        # 1. Base transform: vehicle_link -> arm_base_link
        self.T_base = self._parse_joint_origin(root, base_joint_name)

        # 2. Collect actuated joints in chain order
        self.joint_names = []
        self.joint_types = []       # 'revolute' or 'prismatic'
        self.joint_axes = []        # unit axis (3,)
        self.joint_origins = []     # fixed 4x4 transform
        self.joint_limits = []      # (lower, upper)

        # Also collect the fixed EE joint origin
        self.ee_origin = np.eye(4)

        idx = 1
        while True:
            jname = f"{joint_prefix}_joint{idx}"
            jel = root.find(f"./joint[@name='{jname}']")
            if jel is None:
                break
            self.joint_names.append(jname)
            self.joint_types.append(jel.attrib["type"])
            self.joint_axes.append(self._parse_axis(jel))
            self.joint_origins.append(self._parse_joint_origin_el(jel))
            lim = jel.find("limit")
            self.joint_limits.append(
                (float(lim.attrib["lower"]), float(lim.attrib["upper"]))
            )
            idx += 1

        # EE fixed joint
        ee_name = f"{joint_prefix}_ee_joint"
        ee_el = root.find(f"./joint[@name='{ee_name}']")
        if ee_el is not None:
            self.ee_origin = self._parse_joint_origin_el(ee_el)

        self.n_joints = len(self.joint_names)

    # ---- parsing helpers ----
    @staticmethod
    def _parse_joint_origin(root, joint_name):
        jel = root.find(f"./joint[@name='{joint_name}']")
        return URDFArmChain._parse_joint_origin_el(jel)

    @staticmethod
    def _parse_joint_origin_el(jel):
        origin = jel.find("origin")
        if origin is None:
            return np.eye(4)
        xyz = np.array([float(v) for v in origin.attrib.get("xyz", "0 0 0").split()])
        rpy = np.array([float(v) for v in origin.attrib.get("rpy", "0 0 0").split()])
        return _xyzrpy_to_T(xyz, rpy)

    @staticmethod
    def _parse_axis(jel):
        ax_el = jel.find("axis")
        if ax_el is None:
            return np.array([0.0, 0.0, 1.0])
        return np.array([float(v) for v in ax_el.attrib["xyz"].split()])

    # ---- forward kinematics ----
    def _joint_transform(self, idx, q_val):
        """Return the 4x4 transform contributed by joint `idx` at value `q_val`."""
        T = self.joint_origins[idx].copy()
        ax = self.joint_axes[idx]
        if self.joint_types[idx] == "prismatic":
            T_joint = np.eye(4)
            T_joint[:3, 3] = ax * q_val
        else:  # revolute
            # rotation about `ax` by `q_val`
            T_joint = np.eye(4)
            T_joint[:3, :3] = _axis_angle_to_rot(ax, q_val)
        return T @ T_joint

    def fk_all_frames(self, q):
        """
        Forward kinematics returning origins of every frame (base + each joint
        + EE) in vehicle frame.

        Parameters
        ----------
        q : array-like, shape (n_joints,)

        Returns
        -------
        positions : ndarray, shape (n_joints + 2, 3)
            positions[0] = arm base origin in vehicle frame
            positions[i+1] = frame after joint i
            positions[-1] = EE frame origin
        """
        T = self.T_base.copy()
        pts = [T[:3, 3].copy()]
        for i in range(self.n_joints):
            T = T @ self._joint_transform(i, q[i])
            pts.append(T[:3, 3].copy())
        # EE
        T_ee = T @ self.ee_origin
        pts.append(T_ee[:3, 3].copy())
        return np.array(pts)

    def sample_random_configs(self, n):
        """Return (n, n_joints) array of uniformly sampled joint values."""
        configs = np.empty((n, self.n_joints))
        for i, (lo, hi) in enumerate(self.joint_limits):
            configs[:, i] = np.random.uniform(lo, hi, size=n)
        return configs


def _axis_angle_to_rot(axis, angle):
    """Rodrigues formula for rotation about an arbitrary unit axis."""
    ax = axis / (np.linalg.norm(axis) + 1e-12)
    K = np.array([
        [0, -ax[2], ax[1]],
        [ax[2], 0, -ax[0]],
        [-ax[1], ax[0], 0],
    ])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


# ===========================================================================
# Occupancy grid
# ===========================================================================
class OccupancyGrid:
    """A 2-D boolean occupancy grid on the XY plane."""

    def __init__(self, x_min, x_max, y_min, y_max, resolution):
        self.x_min = x_min
        self.y_min = y_min
        self.res = resolution
        self.nx = int(np.ceil((x_max - x_min) / resolution))
        self.ny = int(np.ceil((y_max - y_min) / resolution))
        self.grid = np.zeros((self.nx, self.ny), dtype=bool)

    def mark_capsule_xy(self, p0, p1, radius):
        """
        Mark all cells within `radius` of the line segment p0-p1
        (XY projection only).  p0, p1 are (2,) or (3,) arrays.
        """
        x0, y0 = float(p0[0]), float(p0[1])
        x1, y1 = float(p1[0]), float(p1[1])

        # Bounding box for this capsule
        bx_min = min(x0, x1) - radius
        bx_max = max(x0, x1) + radius
        by_min = min(y0, y1) - radius
        by_max = max(y0, y1) + radius

        ix_lo = max(0, int((bx_min - self.x_min) / self.res))
        ix_hi = min(self.nx - 1, int((bx_max - self.x_min) / self.res))
        iy_lo = max(0, int((by_min - self.y_min) / self.res))
        iy_hi = min(self.ny - 1, int((by_max - self.y_min) / self.res))

        if ix_lo > ix_hi or iy_lo > iy_hi:
            return

        # Build coordinate arrays for the sub-region
        ix_arr = np.arange(ix_lo, ix_hi + 1)
        iy_arr = np.arange(iy_lo, iy_hi + 1)
        cx = self.x_min + (ix_arr + 0.5) * self.res  # cell centres
        cy = self.y_min + (iy_arr + 0.5) * self.res

        # Meshgrid of cell centres
        CX, CY = np.meshgrid(cx, cy, indexing="ij")  # (len_ix, len_iy)

        # Signed distance from each cell centre to the segment [p0, p1]
        dx, dy = x1 - x0, y1 - y0
        seg_len_sq = dx * dx + dy * dy

        if seg_len_sq < 1e-12:
            # Degenerate segment → just a circle
            dist_sq = (CX - x0) ** 2 + (CY - y0) ** 2
        else:
            t = ((CX - x0) * dx + (CY - y0) * dy) / seg_len_sq
            t = np.clip(t, 0.0, 1.0)
            proj_x = x0 + t * dx
            proj_y = y0 + t * dy
            dist_sq = (CX - proj_x) ** 2 + (CY - proj_y) ** 2

        mask = dist_sq <= radius * radius
        self.grid[ix_lo:ix_hi + 1, iy_lo:iy_hi + 1] |= mask

    def to_xy_arrays(self):
        """Return (x, y) arrays of occupied cell centres."""
        ix, iy = np.where(self.grid)
        x = self.x_min + (ix + 0.5) * self.res
        y = self.y_min + (iy + 0.5) * self.res
        return x, y


# ===========================================================================
# Main computation
# ===========================================================================
def compute_swept_envelope(chain, grid_template_fn, n_samples, link_radius,
                           label="arm"):
    """
    Sample `n_samples` random configs for `chain`, compute FK for all links,
    and mark the capsule occupancy in a fresh grid.
    """
    grid = grid_template_fn()
    configs = chain.sample_random_configs(n_samples)
    report_every = max(1, n_samples // 20)
    for k in range(n_samples):
        if k % report_every == 0:
            print(f"  [{label}] sample {k}/{n_samples}", flush=True)
        pts = chain.fk_all_frames(configs[k])
        # Mark capsule for every consecutive pair of frame origins
        for i in range(len(pts) - 1):
            grid.mark_capsule_xy(pts[i], pts[i + 1], link_radius)
    return grid


def rect_mask(grid, roi):
    """Return a boolean mask (same shape as grid.grid) that is True inside
    the rectangle defined by `roi` dict with x_min/x_max/y_min/y_max."""
    ix = np.arange(grid.nx)
    iy = np.arange(grid.ny)
    cx = grid.x_min + (ix + 0.5) * grid.res
    cy = grid.y_min + (iy + 0.5) * grid.res
    CX, CY = np.meshgrid(cx, cy, indexing="ij")
    return (
        (CX >= roi["x_min"]) & (CX <= roi["x_max"]) &
        (CY >= roi["y_min"]) & (CY <= roi["y_max"])
    )


# ===========================================================================
# Visualisation
# ===========================================================================
def _draw_arm_schematic(ax, chain, q, color, label_prefix):
    """
    Draw a simplified schematic of the arm's XY-plane degrees of freedom.

    The 7-DOF arm (prismatic–revolute–prismatic–revolute×4) is reduced to
    the joints that **produce XY-plane motion**:

      J1 (prismatic, y-axis)  → gray rail from base to max travel extent
      J2, J4, J5, J6 (revolute, z-axis) → open circles with rotation arcs
      Rigid links between adjacent revolute joints → solid line segments

    J3 (prismatic along z = height adjustment) and J7 (end-effector
    orientation) are absorbed into their adjacent links — they contribute
    negligible XY displacement and would only add visual clutter.

    The end-effector is shown as a filled triangle (▼).
    """
    pts = chain.fk_all_frames(q)          # (n_joints+2, 3)

    # --- Key XY positions (simplified grouping) ---
    p_base     = pts[0, :2]     # arm base (fixed mount on vehicle)
    p_shoulder = pts[2, :2]     # J2 shoulder (current slider position)
    p_elbow    = pts[4, :2]     # J4 elbow    (skip J3 vertical)
    p_forearm  = pts[5, :2]     # J5 forearm
    p_wrist    = pts[6, :2]     # J6 wrist
    p_ee       = pts[-1, :2]    # EE (J7 + fixed combined)

    # === 1) J1 prismatic rail ===
    # Draw full rail structure: from base (fixed mount on vehicle)
    # to the maximum J1 travel position.  The URDF has a fixed offset
    # of ~14.8 cm from base_link to J1 origin (then -1 cm for J2 offset,
    # giving ~13.9 cm visible gap); the rail travel adds another 23.4 cm.
    # Drawing only the travel range would leave a visible gap between
    # the base star and the rail.
    q_hi = q.copy()
    q_hi[0] = chain.joint_limits[0][1]
    rail_end = chain.fk_all_frames(q_hi)[2, :2]   # max J1 extent

    # Thick gray bar: base → max J1 travel
    ax.plot([p_base[0], rail_end[0]], [p_base[1], rail_end[1]],
            '-', color='#bbbbbb', lw=7, solid_capstyle='round', zorder=4)
    ax.plot([p_base[0], rail_end[0]], [p_base[1], rail_end[1]],
            '-', color='#888888', lw=7, solid_capstyle='round', zorder=4,
            alpha=0.15)

    # Slider position indicator (small square on the rail)
    ax.plot(p_shoulder[0], p_shoulder[1], 's', color='#555555',
            markersize=6, markeredgecolor='black', markeredgewidth=0.6,
            zorder=4.5)

    # === 2) Arm links — all solid, with white outline for visibility ===
    links = [
        (p_shoulder, p_elbow),    # upper arm  (J2 → J4)
        (p_elbow,    p_forearm),  # forearm    (J4 → J5)
        (p_forearm,  p_wrist),    # lower arm  (J5 → J6)
        (p_wrist,    p_ee),       # hand       (J6 → EE)
    ]
    for i, (a, b) in enumerate(links):
        lbl = label_prefix if i == 0 else None
        ax.plot([a[0], b[0]], [a[1], b[1]], '-', color='white', lw=4.5,
                solid_capstyle='round', zorder=5.9)
        ax.plot([a[0], b[0]], [a[1], b[1]], '-', color=color, lw=3,
                solid_capstyle='round', zorder=6, label=lbl)

    # === 3) Revolute joints — open circles with small rotation arcs ===
    _ARC_RADIUS = 0.018           # metres – size of the rotation indicator
    _ARC_SPAN   = np.pi * 1.6    # radians (~288°) – visible arc sweep
    for p in [p_shoulder, p_elbow, p_forearm, p_wrist]:
        # Open circle marker
        ax.plot(p[0], p[1], 'o', color='white', markersize=7,
                markeredgecolor=color, markeredgewidth=2, zorder=7)
        # Small arc arrow indicating rotation
        theta = np.linspace(np.pi / 6, np.pi / 6 + _ARC_SPAN, 25)
        ax.plot(p[0] + _ARC_RADIUS * np.cos(theta),
                p[1] + _ARC_RADIUS * np.sin(theta),
                '-', color=color, lw=0.9, alpha=0.55, zorder=7.5)

    # === 4) End-effector — filled triangle ===
    ax.plot(p_ee[0], p_ee[1], 'v', color=color, markersize=9,
            markeredgecolor='black', markeredgewidth=0.8, zorder=7.5)


def _add_vehicle_and_bases(ax, T_left, T_right):
    """Draw vehicle body, arm bases, and ROI rectangles."""
    # Vehicle body
    veh = patches.Rectangle((-0.50, -0.25), 1.0, 0.50,
                            linewidth=2, edgecolor="gray",
                            facecolor="lightgray", alpha=0.25,
                            linestyle="--", label="Vehicle body")
    ax.add_patch(veh)

    # Arm bases
    ax.scatter(T_left[0, 3], T_left[1, 3], c="#819CC4", marker="*",
               s=200, zorder=5, label="Left arm base")
    ax.scatter(T_right[0, 3], T_right[1, 3], c="#8CC19A", marker="*",
               s=200, zorder=5, label="Right arm base")


def _add_roi_rects(ax):
    """Draw the ROI upper / lower rectangles."""
    for roi, clr, lbl in [
        (ROI_UPPER, "dimgray", "ROI upper"),
        (ROI_LOWER, "dimgray", "ROI lower"),
    ]:
        r = patches.Rectangle(
            (roi["x_min"], roi["y_min"]),
            roi["x_max"] - roi["x_min"],
            roi["y_max"] - roi["y_min"],
            linewidth=1, edgecolor=clr, facecolor="none",
            linestyle="-", label=lbl, zorder=4,
        )
        ax.add_patch(r)


def _style_ax(ax, title):
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x (vehicle frame, m)")
    ax.set_ylabel("y (vehicle frame, m)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)


def plot_envelopes(grid_L, grid_R, grid_I,
                   grid_I_upper, grid_I_lower,
                   T_left, T_right, save_path=None,
                   chain_L=None, chain_R=None,
                   q_home=None, q_home_L=None, q_home_R=None):
    """Generate a multi-panel figure showing E_L, E_R, E_I and ROI intersections.

    If *chain_L*, *chain_R* and arm configurations are provided, each panel
    draws a simplified schematic of both arms showing only XY-plane DOFs:
      - J1 prismatic slide → gray rail (base to max travel)
      - J2/J4/J5/J6 revolute → open circles with rotation arcs
      - Rigid links → solid lines
      - End-effector → filled triangle

    Parameters *q_home_L* and *q_home_R* allow specifying different
    configurations for the two arms so they can be drawn in clearly
    separated, non-overlapping poses.  If only *q_home* is given, both
    arms use the same configuration.
    """

    # Resolve per-arm configurations
    _q_L = q_home_L if q_home_L is not None else q_home
    _q_R = q_home_R if q_home_R is not None else q_home

    fig, axes = plt.subplots(2, 3, figsize=(24, 16))

    extent = [grid_L.x_min, grid_L.x_min + grid_L.nx * grid_L.res,
              grid_L.y_min, grid_L.y_min + grid_L.ny * grid_L.res]

    # Helper to show a boolean grid as an image
    def _show(ax, grid_data, cmap, title, alpha=0.7):
        # grid_data is (nx, ny); imshow expects (rows=ny, cols=nx)
        ax.imshow(grid_data.T, origin="lower", extent=extent,
                  aspect="equal", cmap=cmap, alpha=alpha, zorder=0)
        _add_vehicle_and_bases(ax, T_left, T_right)
        _add_roi_rects(ax)
        # Draw simplified arm schematic if provided
        if chain_L is not None and chain_R is not None and _q_L is not None:
            _draw_arm_schematic(ax, chain_L, _q_L, color="#2850a0",
                                label_prefix="Left")
            _draw_arm_schematic(ax, chain_R, _q_R, color="#207840",
                                label_prefix="Right")
        _style_ax(ax, title)

    cmap_blue = ListedColormap(["white", "#5b8bd6"])
    cmap_green = ListedColormap(["white", "#6dbd7d"])
    cmap_red = ListedColormap(["white", "#d65b5b"])
    cmap_orange = ListedColormap(["white", "#e8a838"])
    cmap_purple = ListedColormap(["white", "#9b59b6"])

    # Row 0
    _show(axes[0, 0], grid_L.grid.astype(int), cmap_blue,
          f"E_L — Left arm swept envelope\n({np.count_nonzero(grid_L.grid)} cells)")
    axes[0, 0].legend(loc="upper left", fontsize=7)

    _show(axes[0, 1], grid_R.grid.astype(int), cmap_green,
          f"E_R — Right arm swept envelope\n({np.count_nonzero(grid_R.grid)} cells)")
    axes[0, 1].legend(loc="upper left", fontsize=7)

    _show(axes[0, 2], grid_I.grid.astype(int), cmap_red,
          f"E_I = E_L ∩ E_R — Danger zone\n({np.count_nonzero(grid_I.grid)} cells)")
    axes[0, 2].legend(loc="upper left", fontsize=7)

    # Row 1: overlay + ROI intersections
    # Overlay: E_L blue, E_R green, E_I red
    overlay = np.zeros((*grid_L.grid.shape, 4))  # RGBA
    # Left only (not in intersection)
    left_only = grid_L.grid & ~grid_I.grid
    overlay[left_only] = [0.35, 0.55, 0.84, 0.5]
    # Right only (not in intersection)
    right_only = grid_R.grid & ~grid_I.grid
    overlay[right_only] = [0.43, 0.74, 0.49, 0.5]
    # Intersection
    overlay[grid_I.grid] = [0.84, 0.36, 0.36, 0.8]

    ax = axes[1, 0]
    ax.imshow(np.transpose(overlay, (1, 0, 2)), origin="lower", extent=extent,
              aspect="equal", zorder=0)
    _add_vehicle_and_bases(ax, T_left, T_right)
    _add_roi_rects(ax)
    # Draw simplified arm schematic on the overlay panel as well
    if chain_L is not None and chain_R is not None and _q_L is not None:
        _draw_arm_schematic(ax, chain_L, _q_L, color="#2850a0",
                            label_prefix="Left")
        _draw_arm_schematic(ax, chain_R, _q_R, color="#207840",
                            label_prefix="Right")
    _style_ax(ax, "Overlay: E_L (blue) / E_R (green) / E_I (red)")
    # Manual legend entries
    from matplotlib.lines import Line2D
    legend_el = [
        Line2D([0], [0], color="#5b8bd6", lw=6, label="E_L only"),
        Line2D([0], [0], color="#6dbd7d", lw=6, label="E_R only"),
        Line2D([0], [0], color="#d65b5b", lw=6, label="E_I (danger)"),
        Line2D([0], [0], color="#bbb", lw=5, solid_capstyle="round",
               label="Slide rail (J1)"),
        Line2D([0], [0], color="#555", ls="None", marker="s",
               markersize=5, markeredgecolor="black", markeredgewidth=0.5,
               label="Slider position"),
        Line2D([0], [0], color="gray", ls="-", lw=2.5,
               marker="o", markerfacecolor="white", markeredgecolor="gray",
               markersize=5, markeredgewidth=1.5,
               label="Link + revolute joint"),
        Line2D([0], [0], color="gray", ls="None", marker="v",
               markersize=7, label="End-effector"),
    ]
    ax.legend(handles=legend_el, loc="upper left", fontsize=7)

    # ROI upper intersection
    _show(axes[1, 1], grid_I_upper.astype(int), cmap_orange,
          f"E_I ∩ ROI upper\n({np.count_nonzero(grid_I_upper)} cells)")
    axes[1, 1].legend(loc="upper left", fontsize=7)

    # ROI lower intersection
    _show(axes[1, 2], grid_I_lower.astype(int), cmap_purple,
          f"E_I ∩ ROI lower\n({np.count_nonzero(grid_I_lower)} cells)")
    axes[1, 2].legend(loc="upper left", fontsize=7)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    plt.close(fig)


# ===========================================================================
# Entry point
# ===========================================================================
def main():
    np.random.seed(42)

    print("=" * 60)
    print("Danger Zone Computation via Swept Envelope Overlap")
    print("=" * 60)

    # ---- Parse URDF chains ----
    print("\nParsing URDF …")
    chain_L = URDFArmChain(URDF_PATH, "vehicle_to_left_arm", "left_arm")
    chain_R = URDFArmChain(URDF_PATH, "vehicle_to_right_arm", "right_arm")

    print(f"  Left arm : {chain_L.n_joints} joints, base @ "
          f"{chain_L.T_base[:3, 3].tolist()}")
    print(f"  Right arm: {chain_R.n_joints} joints, base @ "
          f"{chain_R.T_base[:3, 3].tolist()}")

    # ---- Grid template factory ----
    def make_grid():
        return OccupancyGrid(GRID_X_MIN, GRID_X_MAX,
                             GRID_Y_MIN, GRID_Y_MAX, GRID_RES)

    os.makedirs(RESULT_DIR, exist_ok=True)
    data_path = os.path.join(RESULT_DIR, "danger_zone_data.npz")

    if os.path.exists(data_path):
        print(f"\nLoading existing danger zone data from {data_path}")
        data = np.load(data_path)
        grid_L = make_grid()
        grid_L.grid = data["grid_L"]
        grid_R = make_grid()
        grid_R.grid = data["grid_R"]
        grid_I = make_grid()
        grid_I.grid = data["grid_I"]
        grid_I_upper = data["grid_I_upper"]
        grid_I_lower = data["grid_I_lower"]
    else:
        # ---- Compute envelopes ----
        print(f"\nSampling {N_SAMPLES} configs per arm …")
        grid_L = compute_swept_envelope(chain_L, make_grid, N_SAMPLES, LINK_RADIUS,
                                        label="Left")
        grid_R = compute_swept_envelope(chain_R, make_grid, N_SAMPLES, LINK_RADIUS,
                                        label="Right")

        n_L = np.count_nonzero(grid_L.grid)
        n_R = np.count_nonzero(grid_R.grid)
        print(f"\n  E_L cells: {n_L}  ({n_L * GRID_RES**2 * 1e4:.1f} cm²)")
        print(f"  E_R cells: {n_R}  ({n_R * GRID_RES**2 * 1e4:.1f} cm²)")

        # ---- Danger zone ----
        grid_I = make_grid()
        grid_I.grid = grid_L.grid & grid_R.grid
        n_I = np.count_nonzero(grid_I.grid)
        print(f"  E_I cells: {n_I}  ({n_I * GRID_RES**2 * 1e4:.1f} cm²)")

        # ---- Intersect with ROI upper / lower ----
        mask_upper = rect_mask(grid_I, ROI_UPPER)
        mask_lower = rect_mask(grid_I, ROI_LOWER)
        grid_I_upper = grid_I.grid & mask_upper
        grid_I_lower = grid_I.grid & mask_lower

        # ---- Save Data ----
        print(f"\nSaving danger zone data → {data_path}")
        np.savez_compressed(
            data_path,
            grid_L=grid_L.grid,
            grid_R=grid_R.grid,
            grid_I=grid_I.grid,
            grid_I_upper=grid_I_upper,
            grid_I_lower=grid_I_lower,
            extent=np.array([GRID_X_MIN, GRID_X_MAX, GRID_Y_MIN, GRID_Y_MAX]),
            res=np.array([GRID_RES])
        )

    n_IU = np.count_nonzero(grid_I_upper)
    n_IL = np.count_nonzero(grid_I_lower)
    print(f"\n  E_I ∩ ROI_upper cells: {n_IU}  ({n_IU * GRID_RES**2 * 1e4:.1f} cm²)")
    print(f"  E_I ∩ ROI_lower cells: {n_IL}  ({n_IL * GRID_RES**2 * 1e4:.1f} cm²)")

    # ---- Home configuration (same seed as build_roi_table) ----
    # 7-DOF home configuration (same as q_seed in build_roi_table.py):
    # [j1_prismatic, j2_revolute, j3_prismatic, j4_revolute, j5_revolute, j6_revolute, j7_revolute]
    # URDF 关节约束限制: 
    # J1(prismatic): [0.001, 0.235] | J2(revolute): [-6.28, 6.28] | J3(prismatic): [0.001, 0.45] 
    # J4(revolute): [-3.25, -0.001] | J5(revolute): [-3.6, -0.001]  | J6(revolute): [0.0, 4.0] 
    # J7(revolute): [-3.14, 3.14]
    q_home = np.array([0.001, -2, 0.23, -2.5, -1, 3.0, 0.0])

    # ---- Visualisation poses ----
    # 在果篮位置的初始姿态（直接使用预定义的 q_home）
    q_vis_L = q_home.copy()
    q_vis_R = q_home.copy()

    # ---- Visualise ----
    fig_path = os.path.join(RESULT_DIR, "danger_zone_envelopes.png")
    print(f"\nGenerating visualisation → {fig_path}")
    plot_envelopes(grid_L, grid_R, grid_I,
                   grid_I_upper, grid_I_lower,
                   chain_L.T_base, chain_R.T_base,
                   save_path=fig_path,
                   chain_L=chain_L, chain_R=chain_R,
                   q_home_L=q_vis_L, q_home_R=q_vis_R)

    print("\nDone ✓")


if __name__ == "__main__":
    main()
