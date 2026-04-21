import os
import os
import pickle
import xml.etree.ElementTree as ET
import numpy as np


# =========================================================
# CENTRAL CONFIG
# 以后你只改这里
# =========================================================
CONFIG = {
    # ---------- file ----------
    "urdf_path": os.path.join(os.path.dirname(os.path.abspath(__file__)), "../urdf/dual_arm_ik_xy_centered.urdf"),

    # ---------- IK chain ----------
    # 用左臂链做 IK；右臂与左臂结构相同，只是基座位姿不同
    "ik_base_link": "left_arm_base_link",
    "ik_tip_link": "left_arm_ee_link",

    # ---------- fixed joints for base poses in vehicle frame ----------
    "left_base_joint": "vehicle_to_left_arm",
    "right_base_joint": "vehicle_to_right_arm",

    # ---------- output ----------
    "roi_table_file": os.path.join(os.path.dirname(os.path.abspath(__file__)), "results/roi_table.pkl"),
    "dual_arm_cost_file": os.path.join(os.path.dirname(os.path.abspath(__file__)), "results/dual_arm_cost.pkl"),

    # ---------- regenerate ----------
    "force_regenerate": True,

    # ---------- speed / quality ----------
    "dist_eps": 0.001,

    "timeout": 0.02,
    "step_xyz": 0.02,
    "step_yaw": 0.50,

    # 即使不把 roll / pitch 当成搜索变量，也仍需要给它们一个“固定名义值 + 小容差”
    "roll_eps": 0.03,
    "pitch_eps": 0.03,
    "yaw_eps": 0.25,

    # 保持在这条机械臂当前更容易出解的名义姿态
    "ee_target_roll": -np.pi / 2.0,
    "ee_target_pitch": 0.0,

    # ---------- scan region in VEHICLE frame ----------
    # 注意：这里扫的是“车体两侧的草莓可能区域”，不是车体中间
    # x 方向：沿车前后方向；扩大至覆盖所有 B1-B6 区域（B3/B6 x_max ≈ 0.60）
    "x_min": -0.50,
    "x_max": 0.50,

    # 左侧果实带（vehicle +y 一侧）；扩大至覆盖 B1/B2/B3 y_max ≈ 0.70
    "left_side_y_min": 0.25,
    "left_side_y_max": 0.65,

    # 右侧果实带（vehicle -y 一侧）；扩大至覆盖 B4/B5/B6 y_min ≈ -0.70
    "right_side_y_min": -0.65,
    "right_side_y_max": -0.25,

    # 这是 vehicle frame 的 z，不是 arm local z
    # z：先只扫一个切片，方便快速调试，先把果实高度抬到比两臂基座高约 0.25~0.30m 的位置
    "z_min": 0.56,
    "z_max": 0.56,

    # yaw：先允许一整圈搜索
    "yaw_min": -np.pi,
    "yaw_max": np.pi,

    "report_every": 50,

    # ---------- nominal seed ----------
    # 7 joints: [prismatic, revolute, prismatic, revolute, revolute, revolute, revolute]
    # URDF 关节约束限制: 
    # J1(prismatic): [0.001, 0.235] | J2(revolute): [-6.28, 6.28] | J3(prismatic): [0.001, 0.45] 
    # J4(revolute): [-3.25, -0.001] | J5(revolute): [-3.6, -0.001]  | J6(revolute): [0.0, 4.0] 
    # J7(revolute): [-3.14, 3.14]
    "q_seed": np.array(
        [0.001, -2, 0.23, -2.5, -1, 3.0, 0.0],
        dtype=float
    ),

    # ---------- joint speed ----------
    # 如果没有真实关节速度上限，就保持 None
    # cost 将表示“最大关节变化量”的相对代价
    "joint_speed": None,
}


# =========================================================
# basic transforms
# =========================================================
def rpy_to_rot(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    rx = np.array([[1, 0, 0],
                   [0, cr, -sr],
                   [0, sr, cr]])
    ry = np.array([[cp, 0, sp],
                   [0, 1, 0],
                   [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0],
                   [sy, cy, 0],
                   [0, 0, 1]])
    return rz @ ry @ rx


def xyzrpy_to_T(xyz, rpy):
    T = np.eye(4)
    T[:3, :3] = rpy_to_rot(rpy[0], rpy[1], rpy[2])
    T[:3, 3] = np.asarray(xyz, dtype=float)
    return T


def pose_xyz_yaw_to_T(x, y, z, yaw, roll, pitch):
    T = np.eye(4)
    T[:3, :3] = rpy_to_rot(roll, pitch, yaw)
    T[:3, 3] = np.array([x, y, z], dtype=float)
    return T


def invert_T(T):
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


# =========================================================
# keys / ranges
# =========================================================
def create_key(x, y, z):
    return f"{x:.3f}_{y:.3f}_{z:.3f}"


def parse_key(key):
    x, y, z = key.split("_")
    return float(x), float(y), float(z)


def angle_key(yaw):
    return f"{yaw:.4f}"


def build_ranges(cfg):
    x_range = np.arange(cfg["x_min"], cfg["x_max"] + 0.5 * cfg["step_xyz"], cfg["step_xyz"])
    z_range = np.arange(cfg["z_min"], cfg["z_max"] + 0.5 * cfg["step_xyz"], cfg["step_xyz"])
    yaw_range = np.arange(cfg["yaw_min"], cfg["yaw_max"] + 0.5 * cfg["step_yaw"], cfg["step_yaw"])

    y_left = np.arange(cfg["left_side_y_min"], cfg["left_side_y_max"] + 0.5 * cfg["step_xyz"], cfg["step_xyz"])
    y_right = np.arange(cfg["right_side_y_min"], cfg["right_side_y_max"] + 0.5 * cfg["step_xyz"], cfg["step_xyz"])

    return x_range, y_left, y_right, z_range, yaw_range


# =========================================================
# URDF parsing
# =========================================================
def load_base_transforms_from_urdf(urdf_path, left_joint_name, right_joint_name):
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    def joint_to_T(joint_name):
        joint = root.find(f"./joint[@name='{joint_name}']")
        if joint is None:
            raise ValueError(f"Joint '{joint_name}' not found in {urdf_path}")

        origin = joint.find("origin")
        if origin is None:
            xyz = [0.0, 0.0, 0.0]
            rpy = [0.0, 0.0, 0.0]
        else:
            xyz = [float(v) for v in origin.attrib.get("xyz", "0 0 0").split()]
            rpy = [float(v) for v in origin.attrib.get("rpy", "0 0 0").split()]

        return xyzrpy_to_T(xyz, rpy)

    return joint_to_T(left_joint_name), joint_to_T(right_joint_name)


# =========================================================
# IK helpers
# =========================================================
def build_center_out_yaw_order(yaw_range):
    yaw_range = np.unique(np.round(np.asarray(yaw_range, dtype=float), 6))
    center = 0.0
    center_idx = int(np.argmin(np.abs(yaw_range - center)))
    center_yaw = float(yaw_range[center_idx])

    others = [float(y) for i, y in enumerate(yaw_range) if i != center_idx]
    others.sort(key=lambda y: (abs(y - center), y))
    return np.array([center_yaw] + others, dtype=float)


def motion_time(q_from, q_to, joint_speed=None):
    q_from = np.asarray(q_from, dtype=float)
    q_to = np.asarray(q_to, dtype=float)
    dq = np.abs(q_to - q_from)

    if joint_speed is None:
        return float(np.max(dq))

    v = np.maximum(np.asarray(joint_speed, dtype=float), 1e-6)
    return float(np.max(dq / v))


def classify_entry(entry, step_yaw):
    angles = np.asarray(entry.get("angles", []), dtype=float)
    if len(angles) == 0:
        return "unreachable"
    return "reachable"


def solve_one_arm_for_vehicle_point(
    solver,
    T_vehicle_from_arm,
    p_vehicle,
    cfg,
):
    """
    对一个 vehicle-frame 点，计算“某一只臂”是否可达。
    由于左右臂结构完全相同，这里统一用左臂 IK 链求，
    只需把 vehicle 点变到对应 arm local frame。
    """
    T_arm_from_vehicle = invert_T(T_vehicle_from_arm)
    p_local_h = T_arm_from_vehicle @ np.array([p_vehicle[0], p_vehicle[1], p_vehicle[2], 1.0], dtype=float)
    x_l, y_l, z_l = p_local_h[:3]

    yaw_range = np.arange(cfg["yaw_min"], cfg["yaw_max"] + 0.5 * cfg["step_yaw"], cfg["step_yaw"])
    ordered_yaws = build_center_out_yaw_order(yaw_range)

    q_home = np.asarray(cfg["q_seed"], dtype=float)
    q_seed = q_home.copy()

    entry = {
        "angles": [],
        "solutions": {},
        "best_yaw": None,
        "best_solution": None,
        "best_time": None,
        "local_xyz": [float(x_l), float(y_l), float(z_l)],
    }

    for yaw in ordered_yaws:
        T_target = pose_xyz_yaw_to_T(
            float(x_l), float(y_l), float(z_l),
            float(yaw),
            cfg["ee_target_roll"],
            cfg["ee_target_pitch"],
        )

        q = solver.ik(
            ee_pose=T_target,
            qinit=q_seed,
            bx=cfg["dist_eps"],
            by=cfg["dist_eps"],
            bz=cfg["dist_eps"],
            brx=cfg["roll_eps"],
            bry=cfg["pitch_eps"],
            brz=cfg["yaw_eps"],
        )

        if q is None:
            continue

        q = np.asarray(q, dtype=float)
        q_seed = q.copy()

        entry["angles"].append(float(yaw))
        entry["solutions"][angle_key(float(yaw))] = q.tolist()

        t = motion_time(q_home, q, cfg["joint_speed"])
        if entry["best_time"] is None or t < entry["best_time"]:
            entry["best_time"] = t
            entry["best_yaw"] = float(yaw)
            entry["best_solution"] = q.tolist()

    return entry


# =========================================================
# ROI generation in VEHICLE frame
# =========================================================
def generate_dual_arm_roi_table(cfg):
    try:
        from tracikpy import TracIKSolver
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "tracikpy is required to build ROI tables. Install dependency/tracikpy or use the bundled prebuilt module."
        ) from exc

    x_range, y_left, y_right, z_range, yaw_range = build_ranges(cfg)

    solver = TracIKSolver(
        cfg["urdf_path"],
        cfg["ik_base_link"],
        cfg["ik_tip_link"],
        timeout=cfg["timeout"],
    )

    T_left, T_right = load_base_transforms_from_urdf(
        cfg["urdf_path"],
        cfg["left_base_joint"],
        cfg["right_base_joint"],
    )

    print("Left base transform:")
    print(T_left)
    print("Right base transform:")
    print(T_right)

    # 只扫车体两侧，不扫中间
    y_all = np.concatenate([y_right, y_left])

    total = len(x_range) * len(y_all) * len(z_range)
    count = 0

    roi = {}
    left_cost = {}
    right_cost = {}

    for x in x_range:
        for y in y_all:
            for z in z_range:
                count += 1
                if count % cfg["report_every"] == 0:
                    print(f"[{count}/{total}] x={x:.3f}, y={y:.3f}, z={z:.3f}", flush=True)

                p_vehicle = np.array([float(x), float(y), float(z)], dtype=float)
                key = create_key(*p_vehicle)

                entry_left = solve_one_arm_for_vehicle_point(
                    solver=solver,
                    T_vehicle_from_arm=T_left,
                    p_vehicle=p_vehicle,
                    cfg=cfg,
                )

                entry_right = solve_one_arm_for_vehicle_point(
                    solver=solver,
                    T_vehicle_from_arm=T_right,
                    p_vehicle=p_vehicle,
                    cfg=cfg,
                )

                roi[key] = {
                    "vehicle_xyz": p_vehicle.tolist(),
                    "left": entry_left,
                    "right": entry_right,
                }

                if entry_left["best_time"] is not None:
                    left_cost[key] = float(entry_left["best_time"])

                if entry_right["best_time"] is not None:
                    right_cost[key] = float(entry_right["best_time"])

    roi_payload = {
        "data": roi,
        "ranges": {
            "x": x_range.tolist(),
            "y_left": y_left.tolist(),
            "y_right": y_right.tolist(),
            "z": z_range.tolist(),
            "yaw": yaw_range.tolist(),
        },
        "meta": {
            "frame": "vehicle_frame",
            "notes": "ROI scanned directly in vehicle frame, on both side fruit bands.",
        }
    }

    with open(cfg["roi_table_file"], "wb") as f:
        pickle.dump(roi_payload, f)

    with open(cfg["dual_arm_cost_file"], "wb") as f:
        pickle.dump(
            {
                "left_cost_table": left_cost,
                "right_cost_table": right_cost,
            },
            f,
        )

    print(f"Saved ROI table to {cfg['roi_table_file']}", flush=True)
    print(f"Saved cost table to {cfg['dual_arm_cost_file']}", flush=True)

    return roi_payload, left_cost, right_cost, T_left, T_right


# =========================================================
# main
# =========================================================
def main():
    if (not CONFIG["force_regenerate"]) and os.path.exists(CONFIG["roi_table_file"]) and os.path.exists(CONFIG["dual_arm_cost_file"]):
        with open(CONFIG["roi_table_file"], "rb") as f:
            roi_payload = pickle.load(f)

        T_left, T_right = load_base_transforms_from_urdf(
            CONFIG["urdf_path"],
            CONFIG["left_base_joint"],
            CONFIG["right_base_joint"],
        )
        print(f"Loaded existing ROI table from {CONFIG['roi_table_file']}", flush=True)
    else:
        roi_payload, left_cost, right_cost, T_left, T_right = generate_dual_arm_roi_table(CONFIG)

if __name__ == "__main__":
    main()