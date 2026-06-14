"""
realcost_planner.py — point-motion 版 Real Cost 双臂采摘规划器（项目当前主力）。

数据来源：ompl/results/point_table/point_table.pkl（由 ompl/point_table.py 生成）。
核心规则：每个点带一个 label
    parallel      —— 存在不进危险区的 safe 路径，两臂可并行
    serial_upper  —— 只能走穿过危险区的 free 路径（上半区），必须串行
    serial_lower  —— 同上（下半区）
    discard       —— 两臂都够不着，丢弃
优化目标：最小化总完工时间（makespan）。

本文件只负责“数据加载 + 启发式 + MILP 建模求解”；所有绘图/保存逻辑在 viz.py。
"""

import os
import sys
import pickle
import random
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import gurobipy as gp
from gurobipy import GRB

# 让本文件无论从哪运行都能找到仓库根目录的 config.py
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import URDF_RELPATH, make_key as _create_key  # noqa: E402


# 默认输入路径
_DEFAULT_URDF = os.path.join(_REPO_ROOT, URDF_RELPATH)
_DEFAULT_POINT_TABLE = os.path.join(_REPO_ROOT, "ompl", "results", "point_table", "point_table.pkl")


def _load_urdf_base_xy(urdf_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 URDF 解析 vehicle_to_left_arm / vehicle_to_right_arm 关节，
    返回 (L_base_xy, R_base_xy) 各为 shape (2,) 的 ndarray。
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    def joint_xy(joint_name: str) -> np.ndarray:
        joint = root.find(f"./joint[@name='{joint_name}']")
        if joint is None:
            raise ValueError(f"Joint '{joint_name}' not found in {urdf_path}")
        origin = joint.find("origin")
        if origin is None:
            return np.zeros(2)
        xyz_str = origin.attrib.get("xyz", "0 0 0").split()
        return np.array([float(xyz_str[0]), float(xyz_str[1])], dtype=float)

    return joint_xy("vehicle_to_left_arm"), joint_xy("vehicle_to_right_arm")


def _best_candidate(cands: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not cands:
        return None
    valid = [c for c in cands if c.get("best_cost") is not None]
    if not valid:
        return None
    return min(valid, key=lambda c: float(c["best_cost"]))


class RealCostPlanner:
    """
    Point-motion 版 Real Cost 规划器。
    - 机械臂基座：从 URDF 解析
    - 数据来源：ompl/results/point_table/point_table.pkl
    - 使用 point-level motion table 中的:
        label            -> parallel / serial_upper / serial_lower
        allowed_arms     -> 允许分配的臂
        must_assign_to   -> 必须分配的臂（若有）
        optimizer_candidates_L / R
    """

    _LABEL_COLOR = {
        "parallel": "#5B71B5",
        "serial_upper": "#E39D3C",
        "serial_lower": "#9B59B6",
        "discard": "#BB5F76",
    }

    def __init__(self,
                 urdf_path: str = _DEFAULT_URDF,
                 point_motion_table_path: str = _DEFAULT_POINT_TABLE,
                 base_operation_time: float = 0.0):
        if not os.path.isfile(urdf_path):
            raise FileNotFoundError(f"URDF not found: {urdf_path}")
        if not os.path.isfile(point_motion_table_path):
            raise FileNotFoundError(f"Point motion table not found: {point_motion_table_path}")

        self.L_base, self.R_base = _load_urdf_base_xy(urdf_path)
        self._urdf_path = urdf_path
        self.point_motion_table_path = point_motion_table_path

        with open(point_motion_table_path, "rb") as f:
            payload = pickle.load(f)
        self.point_payload = payload
        self.point_index: Dict[str, Dict[str, Any]] = {p["key"]: p for p in payload["points"]}

        # 运行时数据（自包含初始化）
        self.base_operation_time = base_operation_time
        self.task_df = None
        self.milp_params = None
        self.heuristic_actions = None
        self.milp_actions = None
        self.improvement = 0

        print(f"[RealCostPlanner:point-motion] point table: {point_motion_table_path}")
        print(f"[RealCostPlanner:point-motion] URDF: {urdf_path}")
        print(f"[RealCostPlanner:point-motion] Left  base: {self.L_base}")
        print(f"[RealCostPlanner:point-motion] Right base: {self.R_base}")
        print(f"[RealCostPlanner:point-motion] total points in table: {len(self.point_index)}")

    # ----------------------------------------------------------
    # 代价：从 point table 取 best 候选的 best_cost
    # ----------------------------------------------------------
    def _compute_processing_time(self, point: np.ndarray, arm: str) -> Optional[float]:
        key = _create_key(float(point[0]), float(point[1]), 0.56)
        p = self.point_index.get(key)
        if p is None or bool(p.get("discard", False)):
            return None
        best = _best_candidate(p.get("optimizer_candidates_L", [])) if arm == "L" else _best_candidate(p.get("optimizer_candidates_R", []))
        if best is None or best.get("best_cost") is None:
            return None
        return 2.0 * float(best["best_cost"]) + self.base_operation_time

    def _build_row_from_point(self, p: Dict[str, Any]) -> Dict[str, Any]:
        allowed_arms = list(p["allowed_arms"])
        must_assign_to = p.get("must_assign_to")
        label = p["label"]

        best_L = _best_candidate(p.get("optimizer_candidates_L", []))
        best_R = _best_candidate(p.get("optimizer_candidates_R", []))

        one_way_L = None if best_L is None else float(best_L["best_cost"])
        one_way_R = None if best_R is None else float(best_R["best_cost"])

        time_to_L = None if one_way_L is None else 2.0 * one_way_L + self.base_operation_time
        time_to_R = None if one_way_R is None else 2.0 * one_way_R + self.base_operation_time

        return {
            "key": p["key"],
            "x": float(p["x"]),
            "y": float(p["y"]),
            "z": float(p["z"]),
            "half": p["half"],
            "label": label,
            "case": p.get("case"),
            "allowed_arms": allowed_arms,
            "must_assign_to": must_assign_to,
            "accessible_by": allowed_arms,  # 向后兼容字段
            "region": label,                # 向后兼容字段，仅供打印；不再表示 B1-B6
            "one_way_cost_L": one_way_L,
            "one_way_cost_R": one_way_R,
            "time_to_L": time_to_L,
            "time_to_R": time_to_R,
            "candidate_mode_L": None if best_L is None else best_L.get("used_mode"),
            "candidate_mode_R": None if best_R is None else best_R.get("used_mode"),
            "best_candidate_L": best_L,
            "best_candidate_R": best_R,
            "safe_count_L": int(p.get("safe_count_L", 0)),
            "safe_count_R": int(p.get("safe_count_R", 0)),
            "free_count_L": int(p.get("free_count_L", 0)),
            "free_count_R": int(p.get("free_count_R", 0)),
        }

    def create_task_dataset(self,
                            points_per_label: Optional[Dict[str, int]] = None,
                            random_seed: int = 0) -> pd.DataFrame:
        """
        从 point_motion_table 中抽样创建任务数据集。
        若 points_per_label 为 None，则加载全部非 discard 点。
        否则按 label 分别随机采样。
        """
        pts = [p for p in self.point_payload["points"] if not bool(p.get("discard", False))]
        if points_per_label is None:
            selected = pts
        else:
            rng = random.Random(random_seed)
            selected = []
            for label, n in points_per_label.items():
                group = [p for p in pts if p["label"] == label]
                if not group:
                    continue
                selected.extend(rng.sample(group, min(int(n), len(group))))

        rows = [self._build_row_from_point(p) for p in selected]
        self.task_df = pd.DataFrame(rows).reset_index(drop=True)
        self._extract_milp_parameters()
        return self.task_df

    def load_task_locations(self, task_locations: List[Dict]) -> pd.DataFrame:
        """
        加载指定点。
        支持：
          {"key": "..."}
          {"x": ..., "y": ..., "z": 0.56}
        """
        selected = []
        for task in task_locations:
            if "key" in task:
                key = str(task["key"])
            else:
                key = _create_key(float(task["x"]), float(task["y"]), float(task.get("z", 0.56)))
            p = self.point_index.get(key)
            if p is None:
                print(f"[RealCostPlanner:point-motion] WARNING: key {key} not found; skipping.")
                continue
            if bool(p.get("discard", False)):
                print(f"[RealCostPlanner:point-motion] WARNING: key {key} is discard; skipping.")
                continue
            selected.append(p)

        rows = [self._build_row_from_point(p) for p in selected]
        self.task_df = pd.DataFrame(rows).reset_index(drop=True)
        self._extract_milp_parameters()
        return self.task_df

    # ----------------------------------------------------------
    # 参数提取：使用 point labels（serial_upper / serial_lower / parallel）
    # ----------------------------------------------------------
    def _extract_milp_parameters(self):
        if self.task_df is None:
            raise ValueError("No task data loaded.")

        parameters = {
            "tasks": [],
            "positions": {},
            "labels": {},
            "halves": {},
            "allowed_arms": {},
            "must_assign_to": {},
            "processing_time": {},
            "serial_upper_set": set(),
            "serial_lower_set": set(),
            "candidate_mode": {},
        }

        for idx, row in self.task_df.iterrows():
            task_id = f"t{idx}"
            parameters["tasks"].append(task_id)
            parameters["positions"][task_id] = (float(row["x"]), float(row["y"]))
            parameters["labels"][task_id] = row["label"]
            parameters["halves"][task_id] = row["half"]
            parameters["allowed_arms"][task_id] = list(row["allowed_arms"])
            parameters["must_assign_to"][task_id] = row["must_assign_to"]

            if row["label"] == "serial_upper":
                parameters["serial_upper_set"].add(task_id)
            elif row["label"] == "serial_lower":
                parameters["serial_lower_set"].add(task_id)

            if "L" in row["allowed_arms"] and pd.notna(row["time_to_L"]):
                parameters["processing_time"][(task_id, "L")] = float(row["time_to_L"])
                parameters["candidate_mode"][(task_id, "L")] = row["candidate_mode_L"]
            if "R" in row["allowed_arms"] and pd.notna(row["time_to_R"]):
                parameters["processing_time"][(task_id, "R")] = float(row["time_to_R"])
                parameters["candidate_mode"][(task_id, "R")] = row["candidate_mode_R"]

        self.milp_params = parameters

    # ----------------------------------------------------------
    # 启发式：phased heuristic
    # ----------------------------------------------------------
    def spatial_order_heuristic(self) -> List[Dict]:
        """
        phased heuristic：

        左臂 phase 顺序：
          1) serial_upper
          2) upper parallel
          3) lower parallel
          4) serial_lower

        右臂 phase 顺序：
          1) serial_lower
          2) lower parallel
          3) upper parallel
          4) serial_upper

        左右臂同时开始；每一步选择“下一任务最早可开始时间”更小的一侧先落子。
        """
        if self.task_df is None:
            raise ValueError("No task data loaded.")

        # ---- 1) 启发式预分配 arm ----
        assignments = {}
        phase_rank = {}

        def _left_rank(row):
            if row["label"] == "serial_upper":
                return 0
            if row["label"] == "parallel" and row["half"] == "upper":
                return 1
            if row["label"] == "parallel" and row["half"] == "lower":
                return 2
            if row["label"] == "serial_lower":
                return 3
            return 99

        def _right_rank(row):
            if row["label"] == "serial_lower":
                return 0
            if row["label"] == "parallel" and row["half"] == "lower":
                return 1
            if row["label"] == "parallel" and row["half"] == "upper":
                return 2
            if row["label"] == "serial_upper":
                return 3
            return 99

        for idx, row in self.task_df.iterrows():
            tid = f"t{idx}"
            allowed = list(row["allowed_arms"])
            must = row["must_assign_to"]

            if must in ("L", "R"):
                arm = must
            elif allowed == ["L"]:
                arm = "L"
            elif allowed == ["R"]:
                arm = "R"
            else:
                # allowed == [L, R]
                l_rank = _left_rank(row)
                r_rank = _right_rank(row)
                if l_rank < r_rank:
                    arm = "L"
                elif r_rank < l_rank:
                    arm = "R"
                else:
                    cL = row["time_to_L"] if pd.notna(row["time_to_L"]) else float("inf")
                    cR = row["time_to_R"] if pd.notna(row["time_to_R"]) else float("inf")
                    arm = "L" if cL <= cR else "R"

            assignments[tid] = arm
            phase_rank[(tid, "L")] = _left_rank(row)
            phase_rank[(tid, "R")] = _right_rank(row)

        # ---- 2) 按 arm 构建 phase 队列 ----
        left_rows = []
        right_rows = []
        for idx, row in self.task_df.iterrows():
            tid = f"t{idx}"
            arm = assignments[tid]
            rec = {
                "tid": tid,
                "idx": idx,
                "arm": arm,
                "x": float(row["x"]),
                "y": float(row["y"]),
                "label": row["label"],
                "half": row["half"],
                "must_assign_to": row["must_assign_to"],
                "time": float(row[f"time_to_{arm}"]),
                "candidate_mode": row[f"candidate_mode_{arm}"],
                "best_candidate": row["best_candidate_L"] if arm == "L" else row["best_candidate_R"],
                "_phase_rank": phase_rank[(tid, arm)],
            }
            if arm == "L":
                left_rows.append(rec)
            else:
                right_rows.append(rec)

        def _sort_rows(rows):
            return sorted(rows, key=lambda r: (r["_phase_rank"], r["time"], r["y"], r["x"]))

        left_rows = _sort_rows(left_rows)
        right_rows = _sort_rows(right_rows)

        # ---- 3) 事件驱动排程：左右臂都从 t=0 开始 ----
        actions = []
        arm_time = {"L": 0.0, "R": 0.0}
        group_busy_until = {"serial_upper": 0.0, "serial_lower": 0.0}
        pos = {"L": 0, "R": 0}

        rows_by_arm = {"L": left_rows, "R": right_rows}

        def _next_ready(arm: str):
            rows = rows_by_arm[arm]
            p = pos[arm]
            if p >= len(rows):
                return None
            rec = rows[p]
            start = arm_time[arm]
            if rec["label"] in ("serial_upper", "serial_lower"):
                start = max(start, group_busy_until[rec["label"]])
            return rec, start

        while True:
            cand_L = _next_ready("L")
            cand_R = _next_ready("R")
            if cand_L is None and cand_R is None:
                break

            choose_arm = None
            if cand_L is None:
                choose_arm = "R"
            elif cand_R is None:
                choose_arm = "L"
            else:
                # 谁更早能开始，先排谁；若相同，先排当前结束更早的一侧
                _, sL = cand_L
                _, sR = cand_R
                if sL < sR:
                    choose_arm = "L"
                elif sR < sL:
                    choose_arm = "R"
                else:
                    choose_arm = "L" if arm_time["L"] <= arm_time["R"] else "R"

            rec, start = _next_ready(choose_arm)
            end = start + rec["time"]

            if rec["label"] in ("serial_upper", "serial_lower"):
                group_busy_until[rec["label"]] = end
            arm_time[choose_arm] = end
            pos[choose_arm] += 1

            actions.append({
                "task": rec["tid"],
                "arm": choose_arm,
                "key": self.task_df.iloc[rec["idx"]]["key"],
                "x": rec["x"],
                "y": rec["y"],
                "start": start,
                "end": end,
                "label": rec["label"],
                "case": self.task_df.iloc[rec["idx"]]["case"],
                "must_assign_to": rec["must_assign_to"],
                "mode": rec["candidate_mode"],
                "ik_index": None if rec["best_candidate"] is None else rec["best_candidate"].get("ik_index"),
            })

        return sorted(actions, key=lambda x: x["start"])

    # ----------------------------------------------------------
    # MILP：使用 serial_upper / serial_lower 串行点集
    # ----------------------------------------------------------
    def build_milp_model(self, warm_start_actions: Optional[List[Dict]] = None) -> gp.Model:
        if self.milp_params is None:
            raise ValueError("No task data loaded.")

        tasks = self.milp_params["tasks"]
        arms = ["L", "R"]
        allowed_arms = self.milp_params["allowed_arms"]
        must_assign_to = self.milp_params["must_assign_to"]
        processing_time = self.milp_params["processing_time"]
        serial_upper_tasks = list(self.milp_params["serial_upper_set"])
        serial_lower_tasks = list(self.milp_params["serial_lower_set"])

        model = gp.Model("DualArmHarvesting_PointMotion")
        model.setParam("OutputFlag", 0)

        x = model.addVars(tasks, arms, vtype=GRB.BINARY, name="x")
        t = model.addVars(tasks, vtype=GRB.CONTINUOUS, name="t")
        T = model.addVar(vtype=GRB.CONTINUOUS, name="T")

        order_vars: Dict[Tuple[str, str, str], gp.Var] = {}
        group_order_vars: Dict[Tuple[str, str, str], gp.Var] = {}

        # 约束 1：每个任务分配给 allowed_arms 中的一只臂
        for i in tasks:
            model.addConstr(gp.quicksum(x[i, a] for a in allowed_arms[i]) == 1,
                            name=f"assign_{i}")
            for a in arms:
                if a not in allowed_arms[i]:
                    model.addConstr(x[i, a] == 0, name=f"forbid_{i}_{a}")

        # 约束 2：must_assign_to
        for i in tasks:
            if must_assign_to[i] in ("L", "R"):
                must = must_assign_to[i]
                model.addConstr(x[i, must] == 1, name=f"must_{i}_{must}")
                for a in arms:
                    if a != must:
                        model.addConstr(x[i, a] == 0, name=f"must_forbid_{i}_{a}")

        M = 10000.0

        # 约束 3：同一手臂任务不可重叠
        for idx_i in range(len(tasks)):
            for idx_j in range(idx_i + 1, len(tasks)):
                i = tasks[idx_i]
                j = tasks[idx_j]
                for a in arms:
                    if a in allowed_arms[i] and a in allowed_arms[j]:
                        o_ij = model.addVar(vtype=GRB.BINARY, name=f"order_{i}_{j}_{a}")
                        o_ji = model.addVar(vtype=GRB.BINARY, name=f"order_{j}_{i}_{a}")
                        order_vars[(i, j, a)] = o_ij
                        order_vars[(j, i, a)] = o_ji
                        model.addConstr(o_ij + o_ji == 1, name=f"order_sum_{i}_{j}_{a}")

                        model.addConstr(
                            t[i] + processing_time[(i, a)]
                            <= t[j] + M * (1 - o_ij) + M * (1 - x[i, a]) + M * (1 - x[j, a]),
                            name=f"seq_{i}_{j}_{a}"
                        )
                        model.addConstr(
                            t[j] + processing_time[(j, a)]
                            <= t[i] + M * (1 - o_ji) + M * (1 - x[i, a]) + M * (1 - x[j, a]),
                            name=f"seq_{j}_{i}_{a}"
                        )

        # 约束 4：serial_upper / serial_lower 组内必须全局串行
        def _add_group_serial_constraints(group_name: str, group_tasks: List[str]):
            for idx_i in range(len(group_tasks)):
                for idx_j in range(idx_i + 1, len(group_tasks)):
                    i = group_tasks[idx_i]
                    j = group_tasks[idx_j]
                    y_ij = model.addVar(vtype=GRB.BINARY, name=f"{group_name}_order_{i}_{j}")
                    y_ji = model.addVar(vtype=GRB.BINARY, name=f"{group_name}_order_{j}_{i}")
                    group_order_vars[(group_name, i, j)] = y_ij
                    group_order_vars[(group_name, j, i)] = y_ji
                    model.addConstr(y_ij + y_ji == 1, name=f"{group_name}_order_sum_{i}_{j}")

                    for a1 in allowed_arms[i]:
                        for a2 in allowed_arms[j]:
                            model.addConstr(
                                t[i] + processing_time[(i, a1)]
                                <= t[j] + M * (1 - y_ij) + M * (1 - x[i, a1]) + M * (1 - x[j, a2]),
                                name=f"{group_name}_seq_{i}_{j}_{a1}_{a2}"
                            )
                    for a1 in allowed_arms[j]:
                        for a2 in allowed_arms[i]:
                            model.addConstr(
                                t[j] + processing_time[(j, a1)]
                                <= t[i] + M * (1 - y_ji) + M * (1 - x[j, a1]) + M * (1 - x[i, a2]),
                                name=f"{group_name}_seq_{j}_{i}_{a1}_{a2}"
                            )

        _add_group_serial_constraints("serial_upper", serial_upper_tasks)
        _add_group_serial_constraints("serial_lower", serial_lower_tasks)

        # 约束 5：总完工时间
        for i in tasks:
            for a in allowed_arms[i]:
                model.addConstr(
                    t[i] + processing_time[(i, a)] <= T + M * (1 - x[i, a]),
                    name=f"makespan_{i}_{a}"
                )

        model.setObjective(T, GRB.MINIMIZE)

        model._varmap = {
            "x": x, "t": t, "T": T,
            "order": order_vars,
            "region_order": group_order_vars,  # 复用旧热启动接口名字
            "tasks": tasks, "arms": arms,
        }

        if warm_start_actions is not None:
            self._set_warm_start(model, warm_start_actions)

        return model

    # ----------------------------------------------------------
    # warm start
    # ----------------------------------------------------------
    def _set_warm_start(self, model: gp.Model, actions: List[Dict]):
        """
        为 point-motion MILP 设置 warm start：
        - 正确设置 serial_upper / serial_lower 的组顺序变量
        """
        try:
            if not hasattr(model, "_varmap") or model._varmap is None:
                raise RuntimeError("Warm start requires model._varmap.")

            varmap = model._varmap
            x = varmap["x"]
            t = varmap["t"]
            T = varmap["T"]
            order_vars = varmap.get("order", {})
            group_order_vars = varmap.get("region_order", {})
            tasks = self.milp_params["tasks"]
            allowed_arms = self.milp_params["allowed_arms"]

            x_warm: Dict[Tuple[str, str], int] = {}
            t_warm: Dict[str, float] = {}

            for tid in tasks:
                for arm in ["L", "R"]:
                    if arm in allowed_arms[tid]:
                        x_warm[(tid, arm)] = 0

            for a in actions:
                tid = a["task"]
                arm = a["arm"]
                x_warm[(tid, arm)] = 1
                t_warm[tid] = float(a["start"])

            T_warm = float(max(a["end"] for a in actions)) if actions else 0.0

            for (tid, arm), value in x_warm.items():
                try:
                    x[tid, arm].Start = int(value)
                except Exception:
                    pass

            for tid in tasks:
                try:
                    t[tid].Start = float(t_warm.get(tid, 0.0))
                except Exception:
                    pass

            try:
                T.Start = T_warm
            except Exception:
                pass

            # 同臂顺序变量
            for arm in ["L", "R"]:
                arm_actions = sorted(
                    [a for a in actions if a["arm"] == arm],
                    key=lambda z: (float(z["start"]), float(z["end"]))
                )
                task_order = [a["task"] for a in arm_actions]
                rank = {tid: i for i, tid in enumerate(task_order)}
                for (i, j, a), var in order_vars.items():
                    if a != arm:
                        continue
                    if i in rank and j in rank:
                        var.Start = 1 if rank[i] < rank[j] else 0

            # serial_upper / serial_lower 组顺序变量
            for group_name in ["serial_upper", "serial_lower"]:
                group_actions = sorted(
                    [a for a in actions if a["label"] == group_name],
                    key=lambda z: (float(z["start"]), float(z["end"]))
                )
                task_order = [a["task"] for a in group_actions]
                rank = {tid: i for i, tid in enumerate(task_order)}
                for key_tuple, var in group_order_vars.items():
                    if len(key_tuple) != 3:
                        continue
                    g, i, j = key_tuple
                    if g != group_name:
                        continue
                    if i in rank and j in rank:
                        var.Start = 1 if rank[i] < rank[j] else 0

            print(f"Warm start values set successfully. Heuristic makespan: {T_warm:.2f}s")

        except Exception as e:
            print(f"Warning: Could not set point-motion warm start values: {e}")

    # ----------------------------------------------------------
    # 求解：MILP 若不优于 heuristic，则回退为 heuristic
    # ----------------------------------------------------------
    def solve_optimization(self,
                           time_limit: int = 1500,
                           heuristic_name: str = "spatial_order"
                           ) -> Tuple[List[Dict], List[Dict], float]:
        if self.task_df is None:
            raise ValueError("No task data loaded.")

        print(f"Running {heuristic_name} heuristic...")
        if heuristic_name != "spatial_order":
            raise ValueError(f"Unknown heuristic: {heuristic_name}")

        self.heuristic_actions = self.spatial_order_heuristic()
        heuristic_makespan = max(action["end"] for action in self.heuristic_actions) if self.heuristic_actions else 0.0
        print(f"Heuristic - Makespan: {heuristic_makespan:.2f}s")

        print("Building MILP model...")
        model = self.build_milp_model(self.heuristic_actions)
        model.setParam("TimeLimit", time_limit)
        model.setParam("MIPFocus", 1)
        model.setParam("OutputFlag", 1)

        print(f"Solving MILP (time limit: {time_limit}s)...")
        model.optimize()

        self.milp_actions = self.heuristic_actions
        self.improvement = 0.0

        if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and model.SolCount > 0:
            x_vals, t_vals = self._extract_solution_vars(model)
            candidate_actions = self._extract_action_sequence(x_vals, t_vals)
            candidate_makespan = max(action["end"] for action in candidate_actions) if candidate_actions else heuristic_makespan

            # 关键保护：
            # point-motion RealCost 模式下，MILP 绝不应比 heuristic 更差；
            # 若当前求解器返回更差解（常见原因是 warm start 未被采纳或 time limit 下只找到较差 incumbent），
            # 则直接保留 heuristic 作为最终结果。
            if candidate_makespan + 1e-9 < heuristic_makespan:
                self.milp_actions = candidate_actions
                self.improvement = ((heuristic_makespan - candidate_makespan) / heuristic_makespan) * 100.0
                print(f"MILP - Makespan: {candidate_makespan:.2f}s")
                print(f"Improvement: {self.improvement:.1f}%")
            else:
                if candidate_makespan > heuristic_makespan + 1e-9:
                    print(
                        f"Warning: MILP incumbent ({candidate_makespan:.2f}s) is worse than heuristic "
                        f"({heuristic_makespan:.2f}s); falling back to heuristic solution."
                    )
                else:
                    print("MILP did not improve over heuristic; keeping heuristic solution.")
                self.milp_actions = self.heuristic_actions
                self.improvement = 0.0
        else:
            print(f"MILP solve failed with status: {model.status}; using heuristic solution.")
            self.milp_actions = self.heuristic_actions
            self.improvement = 0.0

        return self.heuristic_actions, self.milp_actions, self.improvement

    # ----------------------------------------------------------
    # 解提取
    # ----------------------------------------------------------
    def _extract_solution_vars(self, model: gp.Model) -> Tuple[Dict, Dict]:
        x_vals = {}
        t_vals = {}
        for var in model.getVars():
            name = var.VarName
            if name.startswith("x[") and var.X > 0.5:
                task_id, arm = name[2:-1].split(",")
                x_vals[(task_id.strip(), arm.strip())] = 1
            elif name.startswith("t["):
                task_id = name[2:-1].strip()
                t_vals[task_id] = var.X
        return x_vals, t_vals

    def _extract_action_sequence(self, x_vals: Dict, t_vals: Dict) -> List[Dict]:
        actions = []
        for task_id in t_vals:
            row = self.task_df.iloc[int(task_id[1:])]
            arm = 'L' if x_vals.get((task_id, 'L'), 0) > 0.5 else 'R'
            start_time = float(t_vals[task_id])
            end_time = start_time + float(row[f'time_to_{arm}'])
            cand = row["best_candidate_L"] if arm == "L" else row["best_candidate_R"]
            actions.append({
                'task': task_id,
                'arm': arm,
                'key': row["key"],
                'x': float(row['x']),
                'y': float(row['y']),
                'start': start_time,
                'end': end_time,
                'label': row["label"],
                'case': row["case"],
                'must_assign_to': row["must_assign_to"],
                'mode': row["candidate_mode_L"] if arm == "L" else row["candidate_mode_R"],
                'ik_index': None if cand is None else cand.get("ik_index"),
            })
        return sorted(actions, key=lambda a: a['start'])

    # ==========================================================
    # 可视化 / 结果保存
    # ==========================================================
    def animate_solution(self, actions: List[Dict],
                         save_path: Optional[str] = None,
                         title: str = "Point-motion Real Cost Animation") -> animation.FuncAnimation:
        if self.task_df is None:
            raise ValueError("No task data loaded.")

        fig, ax = plt.subplots(figsize=(8.5, 8.5))
        # ROI 上下半区域
        ax.add_patch(plt.Rectangle((-0.5, 0.25), 1.0, 0.40, fill=False,
                                   edgecolor='blue', linestyle='--', linewidth=1.5, label='ROI_upper'))
        ax.add_patch(plt.Rectangle((-0.5, -0.65), 1.0, 0.40, fill=False,
                                   edgecolor='green', linestyle='--', linewidth=1.5, label='ROI_lower'))
        ax.add_patch(plt.Rectangle((-0.5, -0.25), 1.0, 0.50, fill=False,
                                   edgecolor='gray', linestyle=':', linewidth=1.2, label='Vehicle boundary'))

        arm_L, = ax.plot(*self.L_base, 'ks', markersize=10, label='Left Arm Base')
        arm_R, = ax.plot(*self.R_base, 'ko', markersize=10, label='Right Arm Base')
        line_L, = ax.plot([], [], color='blue', lw=3, alpha=0.8, label='Left Arm')
        line_R, = ax.plot([], [], color='red',  lw=3, alpha=0.8, label='Right Arm')
        time_text = ax.text(-0.95, 0.95, '', fontsize=14,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        task_points = {}
        task_states = {}
        for _, row in self.task_df.iterrows():
            color = self._LABEL_COLOR.get(row["label"], "black")
            marker = {"parallel": "o", "serial_upper": "s", "serial_lower": "^", "discard": "x"}.get(row["label"], "o")
            pt, = ax.plot(row["x"], row["y"], marker=marker, color=color, markersize=8, alpha=0.8)
            task_points[f"t{_}"] = pt
            task_states[f"t{_}"] = False

        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_title(title)
        ax.legend(loc='upper right', fontsize=8)

        def init():
            line_L.set_data([], [])
            line_R.set_data([], [])
            time_text.set_text('')
            return [arm_L, arm_R, line_L, line_R, time_text] + list(task_points.values())

        def update(frame):
            t_now = frame * 0.1
            time_text.set_text(f"Time: {t_now:.1f}s")
            line_L.set_data([], [])
            line_R.set_data([], [])

            for act in actions:
                task_id = act['task']
                if act['start'] <= t_now <= act['end']:
                    base = self.L_base if act['arm'] == 'L' else self.R_base
                    if act['arm'] == 'L':
                        line_L.set_data([base[0], act['x']], [base[1], act['y']])
                    else:
                        line_R.set_data([base[0], act['x']], [base[1], act['y']])
                elif t_now > act['end'] and not task_states[task_id]:
                    task_states[task_id] = True
                    task_points[task_id].remove()
                    color = self._LABEL_COLOR.get(act["label"], "black")
                    new_point, = ax.plot(act['x'], act['y'], 'x', color=color, markersize=12, markeredgewidth=3)
                    task_points[task_id] = new_point

            return [arm_L, arm_R, line_L, line_R, time_text] + list(task_points.values())

        max_time = max(act['end'] for act in actions) if actions else 10.0
        frames = int(max_time * 10) + 10
        ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init,
                                      interval=100, repeat=False, blit=False)
        plt.tight_layout()

        if save_path:
            print(f"Saving animation to {save_path}...")
            ani.save(save_path, writer='pillow', fps=10, dpi=100)
            print("Animation saved successfully!")
            plt.close(fig)
        else:
            plt.show()

        return ani

    def plot_gantt_chart(self, actions: List[Dict],
                         save_path: Optional[str] = None,
                         title_suffix: str = "") -> None:
        fig, ax = plt.subplots(figsize=(13, 6))

        left_actions = [a for a in actions if a['arm'] == 'L']
        right_actions = [a for a in actions if a['arm'] == 'R']

        def _draw(arm_actions, y_level):
            for act in arm_actions:
                color = self._LABEL_COLOR.get(act["label"], "gray")
                ax.barh(y_level, act['end'] - act['start'], left=act['start'],
                        height=0.4, color=color, alpha=0.75, edgecolor='black')
                mid_time = 0.5 * (act['start'] + act['end'])
                ax.text(mid_time, y_level,
                        f"{act['task']}\n{act['label']}\n{act['mode']}",
                        ha='center', va='center', fontsize=7, fontweight='bold')

        _draw(left_actions, 0)
        _draw(right_actions, 1)

        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Left Arm (L)', 'Right Arm (R)'])
        ax.set_xlabel('Time (seconds)')
        ax.set_title(f'Point-motion Real Cost Gantt Chart{title_suffix}')
        ax.grid(True, axis='x', alpha=0.3)

        makespan = max(action['end'] for action in actions) if actions else 0
        ax.axvline(x=makespan, color='red', linestyle='--', linewidth=2, alpha=0.8)
        ax.text(makespan, 0.5, f'Makespan: {makespan:.2f}s', rotation=90,
                ha='right', va='center', fontweight='bold', color='red')

        legend_elements = [plt.Rectangle((0, 0), 1, 1,
                                         facecolor=color, alpha=0.75,
                                         edgecolor='black', label=label)
                           for label, color in self._LABEL_COLOR.items()]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gantt chart saved to {save_path}")
            plt.close(fig)
        else:
            plt.show()

    def plot_task_map(self, actions: List[Dict], save_path: Optional[str] = None, title: str = "") -> None:
        fig, ax = plt.subplots(figsize=(9, 9))
        ax.add_patch(plt.Rectangle((-0.5, 0.25), 1.0, 0.40, fill=False, edgecolor="blue", linewidth=1.5, linestyle="--"))
        ax.add_patch(plt.Rectangle((-0.5, -0.65), 1.0, 0.40, fill=False, edgecolor="green", linewidth=1.5, linestyle="--"))
        ax.add_patch(plt.Rectangle((-0.5, -0.25), 1.0, 0.50, fill=False, edgecolor="gray", linewidth=1.2, linestyle=":"))

        for _, row in self.task_df.iterrows():
            color = self._LABEL_COLOR.get(row["label"], "black")
            marker = {"parallel": "o", "serial_upper": "s", "serial_lower": "^", "discard": "x"}.get(row["label"], "o")
            ax.scatter(row["x"], row["y"], c=color, marker=marker, s=80, alpha=0.55)
            txt = ""
            if row["must_assign_to"] in ("L", "R"):
                txt = f'{row["must_assign_to"]}'
            elif row["label"] == "parallel":
                txt = "P"
            elif str(row["label"]).startswith("serial"):
                txt = "S"
            if txt:
                ax.text(row["x"] + 0.010, row["y"] + 0.010, txt, fontsize=8)

        arm_line_color = {"L": "#1f77b4", "R": "#d62728"}
        if actions:
            for act in actions:
                base = self.L_base if act["arm"] == "L" else self.R_base
                ax.plot([base[0], act["x"]], [base[1], act["y"]],
                        color=arm_line_color[act["arm"]], alpha=0.5, linewidth=1.8)
                ax.text(act["x"] + 0.012, act["y"] - 0.018,
                        f'{act["arm"]}:{act["start"]:.1f}-{act["end"]:.1f}',
                        fontsize=6, color=arm_line_color[act["arm"]])

        ax.scatter(self.L_base[0], self.L_base[1], c="black", marker="s", s=140, label="Left base")
        ax.scatter(self.R_base[0], self.R_base[1], c="black", marker="o", s=140, label="Right base")
        ax.set_title(title or "Point-motion RealCost task map")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(-0.55, 0.55)
        ax.set_ylim(-0.75, 0.75)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        legend_items = []
        for label, color in self._LABEL_COLOR.items():
            legend_items.append(plt.Line2D([0], [0], marker='o', color='w',
                                           markerfacecolor=color, markersize=8, label=label))
        legend_items.append(plt.Line2D([0], [0], color=arm_line_color["L"], lw=2, label="assigned to L"))
        legend_items.append(plt.Line2D([0], [0], color=arm_line_color["R"], lw=2, label="assigned to R"))
        ax.legend(handles=legend_items, loc="upper right", fontsize=8)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Task map saved to {save_path}")
            plt.close(fig)
        else:
            plt.show()

    def save_results(self, result_dir: str = "result") -> None:
        if self.heuristic_actions is None:
            raise ValueError("No results to save. Run solve_optimization() first.")

        os.makedirs(result_dir, exist_ok=True)

        heuristic_makespan = max(action["end"] for action in self.heuristic_actions)
        self.animate_solution(self.heuristic_actions,
                              save_path=f"{result_dir}/heuristic_animation.gif")
        self.plot_task_map(self.heuristic_actions,
                           save_path=f"{result_dir}/heuristic_task_map.png",
                           title=f"Heuristic task map (makespan={heuristic_makespan:.2f}s)")
        self.plot_gantt_chart(self.heuristic_actions,
                              save_path=f"{result_dir}/heuristic_gantt.png",
                              title_suffix=f" - Heuristic (Makespan: {heuristic_makespan:.2f}s)")

        if self.milp_actions is not None:
            milp_makespan = max(action["end"] for action in self.milp_actions)
            self.animate_solution(self.milp_actions,
                                  save_path=f"{result_dir}/milp_animation.gif")
            self.plot_task_map(self.milp_actions,
                               save_path=f"{result_dir}/milp_task_map.png",
                               title=f"MILP task map (makespan={milp_makespan:.2f}s)")
            self.plot_gantt_chart(self.milp_actions,
                                  save_path=f"{result_dir}/milp_gantt.png",
                                  title_suffix=f" - MILP Optimized (Makespan: {milp_makespan:.2f}s)")

            with open(f"{result_dir}/comparison_summary.txt", "w") as f:
                f.write("=== Point-motion RealCost Optimization Results ===\n\n")
                f.write(f"Point motion table: {self.point_motion_table_path}\n")
                f.write(f"Total tasks: {len(self.task_df)}\n")
                f.write(f"Label counts: {self.task_df['label'].value_counts().to_dict()}\n\n")
                f.write(f"Heuristic Makespan: {heuristic_makespan:.2f}s\n")
                f.write(f"MILP Makespan: {milp_makespan:.2f}s\n")
                f.write(f"Improvement: {self.improvement:.1f}%\n")

        print(f"Results saved to {result_dir}/ directory")
