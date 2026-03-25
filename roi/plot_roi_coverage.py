import os
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from build_roi_table import CONFIG, classify_entry, load_base_transforms_from_urdf

PLOT_CONFIG = {
    # ---------- vehicle body display ----------
    "body_mode": "auto_from_bases",

    # auto mode margins
    "body_margin_front": 0.15,
    "body_margin_rear": 0.15,
    "body_margin_side": 0.10,

    # manual mode (仅 body_mode="manual" 时生效)
    "body_x_min": -0.60,
    "body_x_max": 0.20,
    "body_y_min": -0.30,
    "body_y_max": 0.30,

    "figure_file": os.path.join(os.path.dirname(os.path.abspath(__file__)), "results/dual_arm_roi_coverage.png"),
}

# =========================================================
# vehicle body drawing
# =========================================================
def compute_vehicle_body_box(plot_cfg, T_left, T_right):
    """
    临时但合理的车体示意：
    - 如果手头没有真实车体尺寸，就至少保证车体框把左右臂基座都包进去
    """
    if plot_cfg["body_mode"] == "manual":
        return (
            plot_cfg["body_x_min"], plot_cfg["body_x_max"],
            plot_cfg["body_y_min"], plot_cfg["body_y_max"]
        )

    # auto_from_bases
    x_left, y_left = T_left[0, 3], T_left[1, 3]
    x_right, y_right = T_right[0, 3], T_right[1, 3]

    x_min = min(x_left, x_right) - plot_cfg["body_margin_rear"]
    x_max = max(x_left, x_right) + plot_cfg["body_margin_front"]
    y_min = min(y_left, y_right) - plot_cfg["body_margin_side"]
    y_max = max(y_left, y_right) + plot_cfg["body_margin_side"]

    return x_min, x_max, y_min, y_max


def draw_vehicle(ax, plot_cfg, T_left, T_right):
    x_min, x_max, y_min, y_max = compute_vehicle_body_box(plot_cfg, T_left, T_right)

    rect = patches.Rectangle(
        (x_min, y_min),
        x_max - x_min,
        y_max - y_min,
        linewidth=2,
        edgecolor="gray",
        facecolor="lightgray",
        alpha=0.25,
        linestyle="--",
        label="Vehicle body",
    )
    ax.add_patch(rect)

    # 前进方向箭头：+x
    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)
    ax.arrow(
        x_center, y_center,
        0.18, 0.0,
        width=0.01,
        head_width=0.05,
        head_length=0.04,
        length_includes_head=True,
        color="black",
        alpha=0.8,
    )
    ax.text(x_center + 0.20, y_center + 0.02, "Forward (+x)", fontsize=10, color="black")

    return x_min, x_max, y_min, y_max


# =========================================================
# plotting
# =========================================================
def plot_dual_arm_roi(roi_payload, cfg, T_left, T_right):
    roi = roi_payload["data"]
    target_z = float(roi_payload["ranges"]["z"][0])

    left_groups = {"unreachable": [], "reachable": []}
    right_groups = {"unreachable": [], "reachable": []}

    summary_groups = {
        "both_unreachable": [],
        "left_only": [],
        "right_only": [],
        "both_reachable": [],
    }

    for key, item in roi.items():
        x, y, z = item["vehicle_xyz"]
        if not np.isclose(z, target_z, atol=1e-6):
            continue

        cls_left = classify_entry(item["left"], cfg["step_yaw"])
        cls_right = classify_entry(item["right"], cfg["step_yaw"])

        left_groups[cls_left].append((x, y))
        right_groups[cls_right].append((x, y))

        left_ok = (cls_left != "unreachable")
        right_ok = (cls_right != "unreachable")

        if left_ok and right_ok:
            summary_groups["both_reachable"].append((x, y))
        elif left_ok and (not right_ok):
            summary_groups["left_only"].append((x, y))
        elif (not left_ok) and right_ok:
            summary_groups["right_only"].append((x, y))
        else:
            summary_groups["both_unreachable"].append((x, y))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # ---------- subplot 1: left arm ----------
    ax = axes[0]
    draw_vehicle(ax, PLOT_CONFIG, T_left, T_right)

    left_color = {
        "unreachable": "#BB5F76",
        "reachable": "#AFC4E4",
    }
    left_label = {
        "unreachable": "Unreachable",
        "reachable": "Reachable",
    }
    for cls, pts in left_groups.items():
        if pts:
            pts = np.asarray(pts)
            ax.scatter(pts[:, 0], pts[:, 1], s=18, c=left_color[cls], label=left_label[cls])

    ax.scatter(T_left[0, 3], T_left[1, 3], c="#819CC4", marker="*", s=180, label="Left arm base")
    ax.scatter(T_right[0, 3], T_right[1, 3], c="#8CC19A", marker="*", s=180, label="Right arm base")
    ax.set_title(f"Left-arm coverage @ z={target_z:.3f}")
    ax.set_xlabel("x in vehicle frame (m)")
    ax.set_ylabel("y in vehicle frame (m)")
    ax.set_aspect("equal")
    ax.grid(True)
    ax.legend(loc="best")

    # ---------- subplot 2: right arm ----------
    ax = axes[1]
    draw_vehicle(ax, PLOT_CONFIG, T_left, T_right)

    right_color = {
        "unreachable": "#BB5F76",
        "reachable": "#BEE4C8",
    }
    for cls, pts in right_groups.items():
        if pts:
            pts = np.asarray(pts)
            ax.scatter(pts[:, 0], pts[:, 1], s=18, c=right_color[cls], label=left_label[cls])

    ax.scatter(T_left[0, 3], T_left[1, 3], c="#819CC4", marker="*", s=180, label="Left arm base")
    ax.scatter(T_right[0, 3], T_right[1, 3], c="#8CC19A", marker="*", s=180, label="Right arm base")
    ax.set_title(f"Right-arm coverage @ z={target_z:.3f}")
    ax.set_xlabel("x in vehicle frame (m)")
    ax.set_ylabel("y in vehicle frame (m)")
    ax.set_aspect("equal")
    ax.grid(True)
    ax.legend(loc="best")

    # ---------- subplot 3: combined accessibility ----------
    ax = axes[2]
    draw_vehicle(ax, PLOT_CONFIG, T_left, T_right)

    summary_color = {
        "both_unreachable": "#BB5F76",
        "left_only": "#AFC4E4",
        "right_only": "#BEE4C8",
        "both_reachable": "#5B71B5",
    }
    summary_label = {
        "both_unreachable": "Both unreachable",
        "left_only": "Left only",
        "right_only": "Right only",
        "both_reachable": "Both reachable",
    }

    for cls, pts in summary_groups.items():
        if pts:
            pts = np.asarray(pts)
            ax.scatter(pts[:, 0], pts[:, 1], s=18, c=summary_color[cls], label=summary_label[cls])

    ax.scatter(T_left[0, 3], T_left[1, 3], c="#819CC4", marker="*", s=180, label="Left arm base")
    ax.scatter(T_right[0, 3], T_right[1, 3], c="#8CC19A", marker="*", s=180, label="Right arm base")
    ax.set_title(f"Combined accessibility @ z={target_z:.3f}")
    ax.set_xlabel("x in vehicle frame (m)")
    ax.set_ylabel("y in vehicle frame (m)")
    ax.set_aspect("equal")
    ax.grid(True)
    ax.legend(loc="best")

    plt.tight_layout()

    if PLOT_CONFIG["figure_file"]:
        os.makedirs(os.path.dirname(PLOT_CONFIG["figure_file"]), exist_ok=True)
        plt.savefig(PLOT_CONFIG["figure_file"], dpi=300, bbox_inches="tight")

    plt.show()

def main():
    roi_file = CONFIG["roi_table_file"]
    cost_file = CONFIG["dual_arm_cost_file"]
    
    if os.path.exists(roi_file) and os.path.exists(cost_file):
        with open(roi_file, "rb") as f:
            roi_payload = pickle.load(f)

        T_left, T_right = load_base_transforms_from_urdf(
            CONFIG["urdf_path"],
            CONFIG["left_base_joint"],
            CONFIG["right_base_joint"],
        )
        print(f"Loaded existing ROI table from {roi_file}", flush=True)
        plot_dual_arm_roi(roi_payload, CONFIG, T_left, T_right)
    else:
        print(f"Error: {roi_file} or {cost_file} not found. Please run build_roi_table.py first.")

if __name__ == "__main__":
    main()
