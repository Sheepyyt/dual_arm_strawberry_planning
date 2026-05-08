
#!/usr/bin/env python3
"""
dual_arm_planner_point_motion.py

基于 point_motion_table_selected_points.pkl 的“轨迹安全感知 Real Cost 模式”。
- 保留 BaselinePlanner 不变（从原 dual_arm_planner.py 直接导入）
- 新的 RealCostPlanner 不再使用 B1-B6 / B2-B5 干涉区逻辑
- 直接读取 point-level motion table，并按以下规则构建优化器输入：

  1) parallel:
     只允许使用 safe 候选
     - both_safe        -> allowed_arms = [L, R]
     - left_safe_only   -> allowed_arms = [L], must_assign_to = L
     - right_safe_only  -> allowed_arms = [R], must_assign_to = R

  2) serial_upper / serial_lower:
     只允许使用 free 候选
     - both_free_no_safe        -> allowed_arms = [L, R]
     - left_free_only_no_safe   -> allowed_arms = [L]
     - right_free_only_no_safe  -> allowed_arms = [R]

  3) discard:
     不进入优化器

说明：
- 对同一点、同一只臂，如果保留了多个候选路径，优化器只使用“该臂最短单程 best_cost 对应”的那一条。
  这是因为在你当前已经确认的点级规则下，同一只臂内部不同候选路径不再引入额外约束；
  因此最短者对优化器来说总是不劣。
"""

import os
import sys
import time
import json
import pickle
import random
from typing import Dict, List, Optional, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

# 与原 planning 模块共存
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from dual_arm_planner import BaselinePlanner, _load_urdf_base_xy  # 保留 baseline 不变


_DEFAULT_URDF = os.path.join(REPO_ROOT, "urdf", "dual_arm_ik_xy_centered.urdf")
_DEFAULT_POINT_TABLE = os.path.join(REPO_ROOT, "ompl", "results", "point_motion_table", "point_motion_table_selected_points.pkl")


def _allowed_arms_from_obj(v: Any) -> List[str]:
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if v is None:
        return []
    s = str(v).strip()
    if not s:
        return []
    if "," in s:
        return [x.strip() for x in s.split(",") if x.strip()]
    return [s]


def _create_key(x: float, y: float, z: float = 0.56) -> str:
    return f"{float(x):.3f}_{float(y):.3f}_{float(z):.3f}"


def _best_candidate(cands: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not cands:
        return None
    valid = [c for c in cands if c.get("best_cost") is not None]
    if not valid:
        return None
    best = min(valid, key=lambda c: float(c["best_cost"]))
    return best


class PointMotionRealCostPlanner:
    """
    point-motion 版 Real Cost Planner。
    """

    def __init__(self,
                 point_motion_table_path: str = _DEFAULT_POINT_TABLE,
                 urdf_path: str = _DEFAULT_URDF,
                 base_operation_time: float = 0.0):
        if not os.path.isfile(point_motion_table_path):
            raise FileNotFoundError(f"Point motion table not found: {point_motion_table_path}")
        if not os.path.isfile(urdf_path):
            raise FileNotFoundError(f"URDF not found: {urdf_path}")

        self.point_motion_table_path = point_motion_table_path
        self.urdf_path = urdf_path
        self.base_operation_time = float(base_operation_time)

        self.L_base, self.R_base = _load_urdf_base_xy(urdf_path)

        with open(point_motion_table_path, "rb") as f:
            payload = pickle.load(f)

        self.point_payload = payload
        self.point_index: Dict[str, Dict[str, Any]] = {p["key"]: p for p in payload["points"]}

        # 运行时变量
        self.task_df: Optional[pd.DataFrame] = None
        self.milp_params: Optional[Dict[str, Any]] = None
        self.heuristic_actions: Optional[List[Dict[str, Any]]] = None
        self.milp_actions: Optional[List[Dict[str, Any]]] = None
        self.improvement: float = 0.0

        print(f"[PointMotionRealCostPlanner] point table: {point_motion_table_path}")
        print(f"[PointMotionRealCostPlanner] URDF: {urdf_path}")
        print(f"[PointMotionRealCostPlanner] Left base  : {self.L_base}")
        print(f"[PointMotionRealCostPlanner] Right base : {self.R_base}")
        print(f"[PointMotionRealCostPlanner] total points in table: {len(self.point_index)}")

    # ------------------------------------------------------------------
    # 数据集构建
    # ------------------------------------------------------------------
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
        从 point motion table 中抽样创建任务数据集。
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
                n_take = min(int(n), len(group))
                selected.extend(rng.sample(group, n_take))

        rows = [self._build_row_from_point(p) for p in selected]
        self.task_df = pd.DataFrame(rows).reset_index(drop=True)
        self._extract_milp_parameters()
        return self.task_df

    def load_task_locations(self, task_locations: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        载入指定点。
        支持：
          {"key": "..."}
          {"x": ..., "y": ..., "z": 0.56}
        """
        selected = []
        for task in task_locations:
            if "key" in task:
                key = str(task["key"])
            else:
                z = float(task.get("z", 0.56))
                key = _create_key(float(task["x"]), float(task["y"]), z)

            if key not in self.point_index:
                print(f"[PointMotionRealCostPlanner] WARNING: key {key} not found; skipping.")
                continue

            p = self.point_index[key]
            if bool(p.get("discard", False)):
                print(f"[PointMotionRealCostPlanner] WARNING: key {key} is discard; skipping.")
                continue
            selected.append(p)

        rows = [self._build_row_from_point(p) for p in selected]
        self.task_df = pd.DataFrame(rows).reset_index(drop=True)
        self._extract_milp_parameters()
        return self.task_df

    # ------------------------------------------------------------------
    # 参数提取
    # ------------------------------------------------------------------
    def _extract_milp_parameters(self):
        if self.task_df is None:
            raise ValueError("No task data loaded.")

        params = {
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
            tid = f"t{idx}"
            params["tasks"].append(tid)
            params["positions"][tid] = (float(row["x"]), float(row["y"]))
            params["labels"][tid] = row["label"]
            params["halves"][tid] = row["half"]
            params["allowed_arms"][tid] = list(row["allowed_arms"])
            params["must_assign_to"][tid] = row["must_assign_to"]

            if row["label"] == "serial_upper":
                params["serial_upper_set"].add(tid)
            elif row["label"] == "serial_lower":
                params["serial_lower_set"].add(tid)

            if "L" in row["allowed_arms"] and pd.notna(row["time_to_L"]):
                params["processing_time"][(tid, "L")] = float(row["time_to_L"])
                params["candidate_mode"][(tid, "L")] = row["candidate_mode_L"]

            if "R" in row["allowed_arms"] and pd.notna(row["time_to_R"]):
                params["processing_time"][(tid, "R")] = float(row["time_to_R"])
                params["candidate_mode"][(tid, "R")] = row["candidate_mode_R"]

        self.milp_params = params

    # ------------------------------------------------------------------
    # 启发式（仅作为 warm start）
    # ------------------------------------------------------------------
    def spatial_order_heuristic(self) -> List[Dict[str, Any]]:
        """
        Phased heuristic（按你确认过的新规则）：

        左臂阶段顺序：
            1) serial_upper 中左臂可达
            2) upper parallel 中左臂可达
            3) lower parallel 中左臂可达
            4) serial_lower 中左臂可达

        右臂阶段顺序：
            1) serial_lower 中右臂可达
            2) lower parallel 中右臂可达
            3) upper parallel 中右臂可达
            4) serial_upper 中右臂可达

        左右臂从 t=0 同时开始工作；启发式只负责产生 warm start。
        对于 allowed_arms = [L, R] 且没有 must_assign_to 的点：
            - 先按 phase preference 选 arm
            - 若 phase rank 相同，再按处理时间较短者选 arm
        """
        if self.task_df is None:
            raise ValueError("No task data loaded.")

        def _phase_rank(row: pd.Series, arm: str) -> int:
            label = str(row["label"])
            half = str(row["half"])
            if arm == "L":
                if label == "serial_upper":
                    return 0
                if label == "parallel" and half == "upper":
                    return 1
                if label == "parallel" and half == "lower":
                    return 2
                if label == "serial_lower":
                    return 3
            else:
                if label == "serial_lower":
                    return 0
                if label == "parallel" and half == "lower":
                    return 1
                if label == "parallel" and half == "upper":
                    return 2
                if label == "serial_upper":
                    return 3
            return 99

        def _ptime(row: pd.Series, arm: str) -> float:
            col = f"time_to_{arm}"
            return float(row[col]) if pd.notna(row[col]) else float("inf")

        # 先给每个点预分配一只臂（仅用于 heuristic warm start）
        assigned = []
        for idx, row in self.task_df.iterrows():
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
                rank_L = _phase_rank(row, "L")
                rank_R = _phase_rank(row, "R")
                if rank_L < rank_R:
                    arm = "L"
                elif rank_R < rank_L:
                    arm = "R"
                else:
                    arm = "L" if _ptime(row, "L") <= _ptime(row, "R") else "R"

            assigned.append((idx, arm))

        left_ids = [idx for idx, arm in assigned if arm == "L"]
        right_ids = [idx for idx, arm in assigned if arm == "R"]

        left_df = self.task_df.loc[left_ids].copy()
        right_df = self.task_df.loc[right_ids].copy()

        # 每只臂内部按 phase rank 排，再按处理时间、x、y 做次排序
        def _sort_df(df: pd.DataFrame, arm: str) -> pd.DataFrame:
            proc_col = f"time_to_{arm}"
            rank = df.apply(lambda r: _phase_rank(r, arm), axis=1)
            return (
                df.assign(_phase_rank=rank)
                  .sort_values(by=["_phase_rank", proc_col, "x", "y"], ascending=[True, True, True, True])
                  .reset_index(drop=False)  # 保留原 task_df index
            )

        left_queue = _sort_df(left_df, "L").to_dict("records")
        right_queue = _sort_df(right_df, "R").to_dict("records")

        actions = []
        arm_time = {"L": 0.0, "R": 0.0}
        group_busy_until = {"serial_upper": 0.0, "serial_lower": 0.0}
        ptr = {"L": 0, "R": 0}
        queues = {"L": left_queue, "R": right_queue}

        def _next_candidate(arm: str):
            q = queues[arm]
            k = ptr[arm]
            if k >= len(q):
                return None
            row = q[k]
            label = str(row["label"])
            ptime = _ptime(pd.Series(row), arm)
            start = arm_time[arm]
            if label in ("serial_upper", "serial_lower"):
                start = max(start, group_busy_until[label])
            return {
                "arm": arm,
                "row": row,
                "start": start,
                "ptime": ptime,
                "phase_rank": int(row["_phase_rank"]),
            }

        while ptr["L"] < len(left_queue) or ptr["R"] < len(right_queue):
            cand_L = _next_candidate("L")
            cand_R = _next_candidate("R")
            cands = [c for c in (cand_L, cand_R) if c is not None]
            if not cands:
                break

            # 事件驱动：谁能更早开始就先排；若并列，再按 phase rank、ptime、arm 名字
            cands.sort(key=lambda c: (c["start"], c["phase_rank"], c["ptime"], c["arm"]))
            c = cands[0]

            arm = c["arm"]
            row = c["row"]
            start = float(c["start"])
            ptime = float(c["ptime"])
            end = start + ptime

            label = str(row["label"])
            if label in ("serial_upper", "serial_lower"):
                group_busy_until[label] = end
            arm_time[arm] = end
            ptr[arm] += 1

            candidate = row["best_candidate_L"] if arm == "L" else row["best_candidate_R"]
            original_idx = int(row["index"])
            actions.append({
                "task": f"t{original_idx}",
                "arm": arm,
                "key": row["key"],
                "x": float(row["x"]),
                "y": float(row["y"]),
                "label": label,
                "start": start,
                "end": end,
                "mode": row["candidate_mode_L"] if arm == "L" else row["candidate_mode_R"],
                "ik_index": None if candidate is None else candidate.get("ik_index"),
            })

        return sorted(actions, key=lambda a: (a["start"], a["arm"], a["task"]))

    # ------------------------------------------------------------------
    # MILP
    # ------------------------------------------------------------------
    def build_milp_model(self, warm_start_actions: Optional[List[Dict[str, Any]]] = None) -> gp.Model:
        if self.milp_params is None:
            raise ValueError("No task data loaded.")

        tasks = self.milp_params["tasks"]
        arms = ["L", "R"]
        allowed_arms = self.milp_params["allowed_arms"]
        must_assign_to = self.milp_params["must_assign_to"]
        processing_time = self.milp_params["processing_time"]
        serial_upper = list(self.milp_params["serial_upper_set"])
        serial_lower = list(self.milp_params["serial_lower_set"])

        model = gp.Model("DualArmHarvesting_PointMotion")
        model.setParam("OutputFlag", 0)

        x = model.addVars(tasks, arms, vtype=GRB.BINARY, name="x")
        t = model.addVars(tasks, vtype=GRB.CONTINUOUS, name="t")
        T = model.addVar(vtype=GRB.CONTINUOUS, name="T")

        order_vars: Dict[Tuple[str, str, str], gp.Var] = {}
        group_order_vars: Dict[Tuple[str, str, str], gp.Var] = {}

        # 每个任务必须被 allowed_arms 中的一只臂执行
        for i in tasks:
            model.addConstr(gp.quicksum(x[i, a] for a in allowed_arms[i]) == 1, name=f"assign_{i}")
            for a in arms:
                if a not in allowed_arms[i]:
                    model.addConstr(x[i, a] == 0, name=f"forbid_{i}_{a}")

        # must_assign_to
        for i in tasks:
            if must_assign_to[i] in ("L", "R"):
                must = must_assign_to[i]
                model.addConstr(x[i, must] == 1, name=f"must_assign_{i}_{must}")
                for a in arms:
                    if a != must:
                        model.addConstr(x[i, a] == 0, name=f"must_forbid_{i}_{a}")

        # 同一手臂任务不能重叠
        M = 10000.0
        for ii in range(len(tasks)):
            for jj in range(ii + 1, len(tasks)):
                i = tasks[ii]
                j = tasks[jj]
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
                            name=f"samearm_{i}_{j}_{a}"
                        )
                        model.addConstr(
                            t[j] + processing_time[(j, a)]
                            <= t[i] + M * (1 - o_ji) + M * (1 - x[i, a]) + M * (1 - x[j, a]),
                            name=f"samearm_{j}_{i}_{a}"
                        )

        # serial_upper 内全局互斥
        def _add_group_serial_constraints(group_name: str, group_tasks: List[str]):
            for ii in range(len(group_tasks)):
                for jj in range(ii + 1, len(group_tasks)):
                    i = group_tasks[ii]
                    j = group_tasks[jj]
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

        _add_group_serial_constraints("serial_upper", serial_upper)
        _add_group_serial_constraints("serial_lower", serial_lower)

        # makespan
        for i in tasks:
            for a in allowed_arms[i]:
                model.addConstr(
                    t[i] + processing_time[(i, a)] <= T + M * (1 - x[i, a]),
                    name=f"makespan_{i}_{a}"
                )

        model.setObjective(T, GRB.MINIMIZE)
        model._varmap = {
            "x": x, "t": t, "T": T,
            "order": order_vars, "group_order": group_order_vars,
            "tasks": tasks, "arms": arms,
        }

        if warm_start_actions is not None:
            self._set_warm_start(model, warm_start_actions)

        return model

    def _set_warm_start(self, model: gp.Model, actions: List[Dict[str, Any]]):
        try:
            varmap = model._varmap
            x = varmap["x"]
            t = varmap["t"]
            T = varmap["T"]
            order_vars = varmap.get("order", {})
            tasks = self.milp_params["tasks"]
            allowed_arms = self.milp_params["allowed_arms"]

            x_warm: Dict[Tuple[str, str], int] = {}
            t_warm: Dict[str, float] = {}
            for tid in tasks:
                for a in ["L", "R"]:
                    if a in allowed_arms[tid]:
                        x_warm[(tid, a)] = 0

            for a in actions:
                tid = a["task"]
                arm = a["arm"]
                x_warm[(tid, arm)] = 1
                t_warm[tid] = float(a["start"])

            T_warm = float(max(a["end"] for a in actions)) if actions else 0.0

            for (tid, arm), value in x_warm.items():
                x[tid, arm].Start = value
            for tid in tasks:
                t[tid].Start = float(t_warm.get(tid, 0.0))
            T.Start = T_warm

            task_index = {tid: idx for idx, tid in enumerate(tasks)}
            for (i, j, arm), v in order_vars.items():
                v.Start = 1 if task_index.get(i, 0) < task_index.get(j, 0) else 0

            print(f"Warm start values set successfully. Heuristic makespan: {T_warm:.2f}s")
        except Exception as e:
            print(f"Warning: Could not set warm start values: {e}")

    # ------------------------------------------------------------------
    # 求解
    # ------------------------------------------------------------------
    def solve_optimization(self, time_limit: int = 30, heuristic_name: str = "spatial_order"):
        if self.task_df is None:
            raise ValueError("No task data loaded.")

        print(f"Running {heuristic_name} heuristic...")
        if heuristic_name != "spatial_order":
            raise ValueError(f"Unknown heuristic: {heuristic_name}")
        self.heuristic_actions = self.spatial_order_heuristic()

        heuristic_makespan = max(a["end"] for a in self.heuristic_actions) if self.heuristic_actions else 0.0
        print(f"Heuristic - Makespan: {heuristic_makespan:.2f}s")

        print("Building MILP model...")
        model = self.build_milp_model(self.heuristic_actions)
        model.setParam("TimeLimit", time_limit)
        model.setParam("MIPFocus", 1)
        model.setParam("OutputFlag", 1)

        print(f"Solving MILP (time limit: {time_limit}s)...")
        model.optimize()

        self.milp_actions = None
        self.improvement = 0.0

        if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and model.SolCount > 0:
            x_vals, t_vals = self._extract_solution_vars(model)
            self.milp_actions = self._extract_action_sequence(x_vals, t_vals)
            milp_makespan = max(a["end"] for a in self.milp_actions) if self.milp_actions else heuristic_makespan
            if heuristic_makespan > 1e-9:
                self.improvement = ((heuristic_makespan - milp_makespan) / heuristic_makespan) * 100.0
            print(f"MILP - Makespan: {milp_makespan:.2f}s")
            print(f"Improvement: {self.improvement:.1f}%")
        else:
            print(f"MILP solve failed with status: {model.status}")
            self.milp_actions = self.heuristic_actions

        return self.heuristic_actions, self.milp_actions, self.improvement

    def _extract_solution_vars(self, model: gp.Model):
        x_vals = {}
        t_vals = {}
        for var in model.getVars():
            name = var.VarName
            if name.startswith("x[") and var.X > 0.5:
                tid, arm = name[2:-1].split(",")
                x_vals[(tid.strip(), arm.strip())] = 1
            elif name.startswith("t["):
                tid = name[2:-1].strip()
                t_vals[tid] = var.X
        return x_vals, t_vals

    def _extract_action_sequence(self, x_vals: Dict[Tuple[str, str], int], t_vals: Dict[str, float]):
        actions = []
        for tid in t_vals:
            row = self.task_df.iloc[int(tid[1:])]
            arm = "L" if x_vals.get((tid, "L"), 0) > 0.5 else "R"
            start = float(t_vals[tid])
            end = start + float(row[f"time_to_{arm}"])
            cand = row["best_candidate_L"] if arm == "L" else row["best_candidate_R"]
            actions.append({
                "task": tid,
                "key": row["key"],
                "arm": arm,
                "x": float(row["x"]),
                "y": float(row["y"]),
                "start": start,
                "end": end,
                "label": row["label"],
                "case": row["case"],
                "must_assign_to": row["must_assign_to"],
                "mode": row["candidate_mode_L"] if arm == "L" else row["candidate_mode_R"],
                "ik_index": None if cand is None else cand.get("ik_index"),
            })
        return sorted(actions, key=lambda a: a["start"])

    # ------------------------------------------------------------------
    # 可视化（不再使用 B1-B6）
    # ------------------------------------------------------------------
    _LABEL_COLOR = {
        "parallel": "#5B71B5",
        "serial_upper": "#E39D3C",
        "serial_lower": "#9B59B6",
        "discard": "#BB5F76",
    }

    def plot_task_map(self, actions: List[Dict[str, Any]], save_path: Optional[str] = None, title: str = ""):
        fig, ax = plt.subplots(figsize=(9, 9))
        # 画 ROI 上下半区域
        ax.add_patch(plt.Rectangle((-0.5, 0.25), 1.0, 0.40, fill=False, edgecolor="blue", linewidth=1.5, linestyle="--"))
        ax.add_patch(plt.Rectangle((-0.5, -0.65), 1.0, 0.40, fill=False, edgecolor="green", linewidth=1.5, linestyle="--"))
        ax.add_patch(plt.Rectangle((-0.5, -0.25), 1.0, 0.50, fill=False, edgecolor="gray", linewidth=1.2, linestyle=":"))

        # 所有任务点（底图）
        for _, row in self.task_df.iterrows():
            color = self._LABEL_COLOR.get(row["label"], "black")
            marker = {"parallel": "o", "serial_upper": "s", "serial_lower": "^", "discard": "x"}.get(row["label"], "o")
            ax.scatter(row["x"], row["y"], c=color, marker=marker, s=80, alpha=0.55)
            # allowed / must_assign 简记
            txt = ""
            if row["must_assign_to"] in ("L", "R"):
                txt = f'{row["must_assign_to"]}'
            elif row["label"] == "parallel":
                txt = "P"
            elif str(row["label"]).startswith("serial"):
                txt = "S"
            if txt:
                ax.text(row["x"] + 0.010, row["y"] + 0.010, txt, fontsize=8)

        # 调度结果覆盖
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
            plt.close(fig)
        else:
            plt.show()

    def plot_gantt_chart(self, actions: List[Dict[str, Any]], save_path: Optional[str] = None, title_suffix: str = ""):
        fig, ax = plt.subplots(figsize=(13, 6))
        left_actions = [a for a in actions if a["arm"] == "L"]
        right_actions = [a for a in actions if a["arm"] == "R"]

        def _draw_arm(arm_actions, y_level):
            for a in arm_actions:
                color = self._LABEL_COLOR.get(a["label"], "gray")
                ax.barh(y_level, a["end"] - a["start"], left=a["start"],
                        height=0.4, color=color, alpha=0.75, edgecolor="black")
                mid = 0.5 * (a["start"] + a["end"])
                ax.text(mid, y_level, f'{a["task"]}\n{a["label"]}\n{a["mode"]}',
                        ha="center", va="center", fontsize=7)

        _draw_arm(left_actions, 0)
        _draw_arm(right_actions, 1)

        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Left Arm", "Right Arm"])
        ax.set_xlabel("Time (seconds)")
        ax.set_title(f"Point-motion RealCost Gantt Chart{title_suffix}")
        ax.grid(True, axis="x", alpha=0.3)

        makespan = max(a["end"] for a in actions) if actions else 0.0
        ax.axvline(x=makespan, color="red", linestyle="--", linewidth=2, alpha=0.8)
        ax.text(makespan, 0.5, f'Makespan: {makespan:.2f}s', rotation=90,
                ha='right', va='center', fontweight='bold', color='red')

        legend_items = []
        for label, color in self._LABEL_COLOR.items():
            legend_items.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="black", label=label))
        ax.legend(handles=legend_items, loc="upper right", fontsize=8)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

    def save_results(self, result_dir: str = "results/real_cost_point_motion"):
        if self.heuristic_actions is None:
            raise ValueError("No results to save. Run solve_optimization() first.")
        os.makedirs(result_dir, exist_ok=True)

        heuristic_makespan = max(a["end"] for a in self.heuristic_actions) if self.heuristic_actions else 0.0
        self.plot_task_map(self.heuristic_actions, save_path=os.path.join(result_dir, "heuristic_task_map.png"),
                           title=f"Heuristic task map (makespan={heuristic_makespan:.2f}s)")
        self.plot_gantt_chart(self.heuristic_actions, save_path=os.path.join(result_dir, "heuristic_gantt.png"),
                              title_suffix=f" - Heuristic (Makespan: {heuristic_makespan:.2f}s)")

        milp_makespan = None
        if self.milp_actions is not None:
            milp_makespan = max(a["end"] for a in self.milp_actions) if self.milp_actions else 0.0
            self.plot_task_map(self.milp_actions, save_path=os.path.join(result_dir, "milp_task_map.png"),
                               title=f"MILP task map (makespan={milp_makespan:.2f}s)")
            self.plot_gantt_chart(self.milp_actions, save_path=os.path.join(result_dir, "milp_gantt.png"),
                                  title_suffix=f" - MILP (Makespan: {milp_makespan:.2f}s)")

        summary = {
            "point_motion_table_path": self.point_motion_table_path,
            "urdf_path": self.urdf_path,
            "total_tasks": 0 if self.task_df is None else int(len(self.task_df)),
            "label_counts": {} if self.task_df is None else self.task_df["label"].value_counts().to_dict(),
            "heuristic_makespan": heuristic_makespan,
            "milp_makespan": milp_makespan,
            "improvement_percent": self.improvement,
        }
        with open(os.path.join(result_dir, "comparison_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        with open(os.path.join(result_dir, "comparison_summary.txt"), "w", encoding="utf-8") as f:
            f.write("=== Point-motion RealCost Optimization Results ===\n\n")
            f.write(f"Point motion table: {self.point_motion_table_path}\n")
            f.write(f"Total tasks: {summary['total_tasks']}\n")
            f.write(f"Label counts: {summary['label_counts']}\n\n")
            f.write(f"Heuristic Makespan: {heuristic_makespan:.2f}s\n")
            if milp_makespan is not None:
                f.write(f"MILP Makespan: {milp_makespan:.2f}s\n")
                f.write(f"Improvement: {self.improvement:.1f}%\n")
            else:
                f.write("MILP Makespan: (fallback to heuristic)\n")

        print(f"Results saved to {result_dir}")


# 为了方便替换旧 RealCost 模式，这里提供相同类名
RealCostPlanner = PointMotionRealCostPlanner

__all__ = ["BaselinePlanner", "PointMotionRealCostPlanner", "RealCostPlanner"]
