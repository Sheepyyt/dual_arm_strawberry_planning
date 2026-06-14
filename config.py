"""
config.py — 全项目共享的几何 / 采摘区域常数（单一事实来源 / single source of truth）

背景：
    原本这些“魔法数字”（ROI 扫描范围、危险区分辨率、连杆半径、上下半区边界、
    机械臂基座关节名、采摘高度 z、点的 key 格式）散落在
        roi/build_roi_table.py、roi/compute_danger_zone.py、roi/build_points.py、
        ompl/plan.py、ompl/point_table.py、planning/*.py
    每个文件各抄一份，改一个就得手动同步好几处，极易改漏。

    现在统一放在这里。以后要调 ROI 范围、危险区分辨率、半区边界等，**只改这一个文件**。

注意：
    本文件保持“零重依赖”（只用标准库 math），因此 roi / ompl / planning
    三个不同 conda 环境都能安全 import。
"""

import math

# ===========================================================
# URDF / 机械臂基座关节名
# ===========================================================
# 相对仓库根目录的 URDF 路径；流水线全程只用这一个 URDF。
URDF_RELPATH = "urdf/dual_arm_ik_xy_centered.urdf"

# 在 vehicle（车体）坐标系下，左右臂基座对应的固定关节名
LEFT_BASE_JOINT = "vehicle_to_left_arm"
RIGHT_BASE_JOINT = "vehicle_to_right_arm"

# ===========================================================
# ROI（感兴趣区域 / 车体两侧的草莓带）扫描范围，单位：米
# ===========================================================
ROI_X_MIN, ROI_X_MAX = -0.50, 0.50

# 上半区（vehicle +y 一侧）与下半区（vehicle -y 一侧）的矩形范围
ROI_UPPER = {"x_min": ROI_X_MIN, "x_max": ROI_X_MAX, "y_min": 0.25, "y_max": 0.65}
ROI_LOWER = {"x_min": ROI_X_MIN, "x_max": ROI_X_MAX, "y_min": -0.65, "y_max": -0.25}

# 采摘高度（当前只扫一个 z 切片，便于快速调试）
Z_DEFAULT = 0.56

# ===========================================================
# 上半区 / 下半区 的判定边界
# ===========================================================
HALF_UPPER_Y_MIN = 0.25     # y >= 此值 → upper（上半区）
HALF_LOWER_Y_MAX = -0.25    # y <= 此值 → lower（下半区）


def half_from_y(y: float) -> str:
    """根据点的 y 坐标判定它属于上半区还是下半区（二分版本：以 0 为界）。"""
    return "upper" if float(y) >= HALF_UPPER_Y_MIN else "lower"


# ===========================================================
# 危险区占用栅格 / 连杆碰撞厚度
# ===========================================================
GRID_RES = 0.005      # 危险区占用栅格分辨率，单位：米
LINK_RADIUS = 0.03    # 连杆胶囊半径（碰撞厚度），单位：米

# ===========================================================
# 点的字典 key 统一格式："x_y_z"，各保留 3 位小数
# ===========================================================
def make_key(x: float, y: float, z: float = Z_DEFAULT) -> str:
    """把一个三维点变成统一字符串 key，例如 (0.0, 0.3, 0.56) -> '0.000_0.300_0.560'。"""
    return f"{float(x):.3f}_{float(y):.3f}_{float(z):.3f}"


# yaw 全圈搜索范围（部分 roi 脚本会用到）
YAW_MIN = -math.pi
YAW_MAX = math.pi
