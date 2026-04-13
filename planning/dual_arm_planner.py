import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import os
import time
import xml.etree.ElementTree as ET
import pickle
from typing import List, Dict, Tuple, Optional, Union


# ============================================================
# Abstract Base Class: DualArmPlannerCore
# ============================================================

class DualArmPlannerCore:
    """
    通用双臂采摘规划器基类。
    封装 MILP 建模、优化求解、热启动、可视化等全部共用逻辑。
    L_base / R_base 与 _compute_processing_time 由子类决定。
    """

    # 子类必须在调用 super().__init__() 之前设置 L_base / R_base
    L_base: np.ndarray
    R_base: np.ndarray

    # 回篮/放篮的安全末端点位 (Home pose in Cartesian space)
    # 子类在 __init__ 中设置具体的数值
    L_home: np.ndarray
    R_home: np.ndarray

    def __init__(self,
                 regions_config: Optional[List[Dict]] = None,
                 base_operation_time: float = 0.0):
        """
        参数:
            regions_config: 采摘区域配置列表；为 None 时使用默认 B1-B6 六区域。
            base_operation_time: 单个草莓的固定处理时间（采摘 + 放置等）。
        """
        self.base_operation_time = base_operation_time

        if regions_config is None:
            self.regions = [
                {"center": (-0.3,  0.45), "width": 0.3, "height": 0.3, "name": "B1", "arm_access": ["L"]},
                {"center": ( 0.0,  0.45), "width": 0.3, "height": 0.3, "name": "B2", "arm_access": ["L", "R"]},
                {"center": ( 0.3,  0.45), "width": 0.3, "height": 0.3, "name": "B3", "arm_access": ["R"]},
                {"center": (-0.3, -0.45), "width": 0.3, "height": 0.3, "name": "B4", "arm_access": ["L"]},
                {"center": ( 0.0, -0.45), "width": 0.3, "height": 0.3, "name": "B5", "arm_access": ["L", "R"]},
                {"center": ( 0.3, -0.45), "width": 0.3, "height": 0.3, "name": "B6", "arm_access": ["R"]},
            ]
        else:
            self.regions = regions_config

        self.colors = ["red", "orange", "green", "blue", "purple", "brown"]
        self.interference_regions = ["B2", "B5"]

        # 运行时数据
        self.task_df = None
        self.milp_params = None
        self.heuristic_actions = None
        self.milp_actions = None
        self.improvement = 0

    # ----------------------------------------------------------
    # 抽象方法：子类必须实现
    # ----------------------------------------------------------
    def _compute_processing_time(self, point: np.ndarray, arm: str) -> Optional[float]:
        """
        计算采摘总处理时间 = 2 * 单程移动时间 + 固定处理时间。
        子类须根据自身代价来源（欧氏距离 / cost table）覆写此方法。
        """
        raise NotImplementedError("Subclasses must implement _compute_processing_time")

    # ----------------------------------------------------------
    # 参数提取（共用）
    # ----------------------------------------------------------
    def _extract_milp_parameters(self):
        parameters = {
            "tasks": [],
            "positions": {},
            "regions": {},
            "accessibility": {},
            "processing_time": {},
            "interference_set": set()
        }
        for idx, row in self.task_df.iterrows():
            task_id = f"t{idx}"
            parameters["tasks"].append(task_id)
            parameters["positions"][task_id] = (row["x"], row["y"])
            parameters["regions"][task_id] = row["region"]
            parameters["accessibility"][task_id] = row["accessible_by"]
            if row["region"] in self.interference_regions:
                parameters["interference_set"].add(task_id)
            if "L" in row["accessible_by"]:
                parameters["processing_time"][(task_id, "L")] = row["time_to_L"]
            if "R" in row["accessible_by"]:
                parameters["processing_time"][(task_id, "R")] = row["time_to_R"]
        self.milp_params = parameters

    # ----------------------------------------------------------
    # 调度辅助：单区域任务调度（基类共用）
    # ----------------------------------------------------------
    def _schedule_region_df(self,
                            arm: str,
                            region: str,
                            region_tasks: pd.DataFrame,
                            start_time: float,
                            region_busy_until: Dict[str, float]
                            ) -> Tuple[List[Dict], float]:
        """
        调度给定 DataFrame 中属于 arm 的任务，返回 (actions, new_start_time)。
        region_tasks 已预先过滤为"属于该臂、属于该区域"的子集。
        """
        if region_tasks.empty:
            return [], start_time

        region_tasks = region_tasks.copy()
        region_tasks["proc_time"] = region_tasks[f"time_to_{arm}"]
        region_tasks = region_tasks.sort_values("proc_time")
        current_time = start_time

        region_actions = []
        for idx, row in region_tasks.iterrows():
            if region in self.interference_regions:
                actual_start = max(current_time, region_busy_until.get(region, 0.0))
                region_busy_until[region] = actual_start + row[f"time_to_{arm}"]
            else:
                actual_start = current_time
            end_time = actual_start + row[f"time_to_{arm}"]

            region_actions.append({
                "task": f"t{idx}",
                "arm": arm,
                "x": row["x"],
                "y": row["y"],
                "start": actual_start,
                "end": end_time
            })
            current_time = end_time

        return region_actions, current_time

    # ----------------------------------------------------------
    # 优化入口（共用）
    # ----------------------------------------------------------
    def solve_optimization(self,
                           time_limit: int = 30,
                           heuristic_name: str = "spatial_order"
                           ) -> Tuple[List[Dict], List[Dict], float]:
        """
        第一步运行启发式，第二步构建并求解 MILP。
        """
        if self.task_df is None:
            raise ValueError("No task data loaded.")

        print(f"Running {heuristic_name} heuristic...")
        if heuristic_name == "spatial_order":
            self.heuristic_actions = self.spatial_order_heuristic()
        else:
            raise ValueError(f"Unknown heuristic: {heuristic_name}")

        heuristic_makespan = max(action["end"] for action in self.heuristic_actions)
        print(f"Heuristic - Makespan: {heuristic_makespan:.2f}s")

        print("Building MILP model...")
        model = self.build_milp_model(self.heuristic_actions)
        model.setParam("TimeLimit", time_limit)
        model.setParam("MIPFocus", 1)
        model.setParam("OutputFlag", 1)

        print(f"Solving MILP (time limit: {time_limit}s)...")
        model.optimize()

        self.milp_actions = None
        self.improvement = 0

        if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
            if model.SolCount > 0:
                x_vals, t_vals = self._extract_solution_vars(model)
                self.milp_actions = self._extract_action_sequence(x_vals, t_vals)
                milp_makespan = max(action["end"] for action in self.milp_actions)
                self.improvement = ((heuristic_makespan - milp_makespan) / heuristic_makespan) * 100
                print(f"MILP - Makespan: {milp_makespan:.2f}s")
                print(f"Improvement: {self.improvement:.1f}%")
            else:
                print("MILP could not find feasible solution.")
                self.milp_actions = self.heuristic_actions
        else:
            print(f"MILP solve failed with status: {model.status}")
            self.milp_actions = self.heuristic_actions

        return self.heuristic_actions, self.milp_actions, self.improvement

    # ----------------------------------------------------------
    # 启发式（基类提供 Baseline 默认实现，RealCostPlanner 会覆写）
    # ----------------------------------------------------------
    def spatial_order_heuristic(self) -> List[Dict]:
        """
        Baseline 空间顺序启发式（默认实现）。
        左臂: B2 → B1 → B4
        右臂: B5 → B6 → B3
        子类如需不同行为请覆写此方法。
        """
        if self.task_df is None:
            raise ValueError("No task data loaded.")

        actions = []
        arm_times = {'L': 0.0, 'R': 0.0}
        region_busy_until = {'B2': 0.0, 'B5': 0.0}

        left_arm_order  = ['B2', 'B1', 'B4']
        right_arm_order = ['B5', 'B6', 'B3']

        def get_region_df(arm, region):
            return self.task_df[
                (self.task_df['region'] == region) &
                (self.task_df['accessible_by'].apply(lambda x: arm in x))
            ]

        print("Left arm sequence: B2 -> B1 -> B4")
        for region in left_arm_order:
            rdf = get_region_df('L', region)
            region_actions, arm_times['L'] = self._schedule_region_df(
                'L', region, rdf, arm_times['L'], region_busy_until)
            actions.extend(region_actions)

        print("Right arm sequence: B5 -> B6 -> B3")
        for region in right_arm_order:
            rdf = get_region_df('R', region)
            region_actions, arm_times['R'] = self._schedule_region_df(
                'R', region, rdf, arm_times['R'], region_busy_until)
            actions.extend(region_actions)

        return sorted(actions, key=lambda x: x["start"])

    # ----------------------------------------------------------
    # MILP 模型构建（共用）
    # ----------------------------------------------------------
    def build_milp_model(self, warm_start_actions: Optional[List[Dict]] = None) -> gp.Model:
        """构建 MILP 模型并可选设置热启动。"""
        if self.milp_params is None:
            raise ValueError("No task data loaded. Call create_task_dataset() or load_task_locations() first.")

        tasks = self.milp_params["tasks"]
        arms = ["L", "R"]
        interference_tasks = self.milp_params["interference_set"]
        processing_time = self.milp_params["processing_time"]
        accessibility = self.milp_params["accessibility"]

        model = gp.Model("DualArmHarvesting")
        model.setParam('OutputFlag', 0)

        x = model.addVars(tasks, arms, vtype=GRB.BINARY, name="x")
        t = model.addVars(tasks, vtype=GRB.CONTINUOUS, name="t")
        T = model.addVar(vtype=GRB.CONTINUOUS, name="T")

        order_vars: Dict[Tuple[str, str, str], gp.Var] = {}
        region_order_vars: Dict[Tuple[str, str], gp.Var] = {}

        # 约束 1：每个任务只分配给一只手臂
        for i in tasks:
            model.addConstr(gp.quicksum(x[i, a] for a in accessibility[i]) == 1,
                            name=f"assign_{i}")

        # 约束 2：遵守可达性
        for i in tasks:
            for a in arms:
                if a not in accessibility[i]:
                    model.addConstr(x[i, a] == 0, name=f"reach_{i}_{a}")

        # 约束 3：同一手臂的顺序约束
        for idx_i in range(len(tasks)):
            for idx_j in range(idx_i + 1, len(tasks)):
                i = tasks[idx_i]
                j = tasks[idx_j]
                for a in arms:
                    if a in accessibility[i] and a in accessibility[j]:
                        o_ij = model.addVar(vtype=GRB.BINARY, name=f"order_{i}_{j}_{a}")
                        o_ji = model.addVar(vtype=GRB.BINARY, name=f"order_{j}_{i}_{a}")
                        order_vars[(i, j, a)] = o_ij
                        order_vars[(j, i, a)] = o_ji
                        model.addConstr(o_ij + o_ji == 1, name=f"order_sum_{i}_{j}_{a}")
                        M = 1000
                        model.addConstr(
                            t[i] + processing_time[i, a]
                            <= t[j] + M * (1 - o_ij) + M * (1 - x[i, a]) + M * (1 - x[j, a]),
                            name=f"seq_{i}_{j}_{a}")
                        model.addConstr(
                            t[j] + processing_time[j, a]
                            <= t[i] + M * (1 - o_ji) + M * (1 - x[i, a]) + M * (1 - x[j, a]),
                            name=f"seq_{j}_{i}_{a}")

        # 约束 4：干涉区内的任务必须串行
        b2_tasks = [i for i in interference_tasks if self.milp_params["regions"][i] == "B2"]
        b5_tasks = [i for i in interference_tasks if self.milp_params["regions"][i] == "B5"]

        for region_tasks in [b2_tasks, b5_tasks]:
            for idx_i in range(len(region_tasks)):
                for idx_j in range(idx_i + 1, len(region_tasks)):
                    i = region_tasks[idx_i]
                    j = region_tasks[idx_j]
                    y_ij = model.addVar(vtype=GRB.BINARY, name=f"region_order_{i}_{j}")
                    y_ji = model.addVar(vtype=GRB.BINARY, name=f"region_order_{j}_{i}")
                    region_order_vars[(i, j)] = y_ij
                    region_order_vars[(j, i)] = y_ji
                    model.addConstr(y_ij + y_ji == 1, name=f"region_order_sum_{i}_{j}")
                    M = 1000
                    for a1 in accessibility[i]:
                        for a2 in accessibility[j]:
                            model.addConstr(
                                t[i] + processing_time[i, a1]
                                <= t[j] + M * (1 - y_ij) + M * (1 - x[i, a1]) + M * (1 - x[j, a2]),
                                name=f"region_seq_{i}_{j}_{a1}_{a2}")
                    for a1 in accessibility[j]:
                        for a2 in accessibility[i]:
                            model.addConstr(
                                t[j] + processing_time[j, a1]
                                <= t[i] + M * (1 - y_ji) + M * (1 - x[j, a1]) + M * (1 - x[i, a2]),
                                name=f"region_seq_{j}_{i}_{a1}_{a2}")

        # 约束 5：总完工时间下界
        for i in tasks:
            for a in accessibility[i]:
                model.addConstr(
                    t[i] + processing_time[i, a] <= T + 1000 * (1 - x[i, a]),
                    name=f"makespan_{i}_{a}")

        # 目标：最小化总完工时间
        model.setObjective(T, GRB.MINIMIZE)

        model._varmap = {
            "x": x, "t": t, "T": T,
            "order": order_vars,
            "region_order": region_order_vars,
            "tasks": tasks, "arms": arms,
        }

        if warm_start_actions is not None:
            self._set_warm_start(model, warm_start_actions)

        return model

    # ----------------------------------------------------------
    # 解提取（共用）
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
            start_time = t_vals[task_id]
            end_time = start_time + row[f'time_to_{arm}']
            actions.append({
                'task': task_id, 'arm': arm,
                'x': row['x'], 'y': row['y'],
                'start': start_time, 'end': end_time
            })
        return sorted(actions, key=lambda a: a['start'])

    # ----------------------------------------------------------
    # 热启动（共用）
    # ----------------------------------------------------------
    def _set_warm_start(self, model: gp.Model, actions: List[Dict]):
        try:
            if not hasattr(model, "_varmap") or model._varmap is None:
                raise RuntimeError("Warm start requires model._varmap.")
            varmap = model._varmap
            x = varmap["x"]
            t = varmap["t"]
            T = varmap["T"]
            order_vars = varmap.get("order", {})
            region_order_vars = varmap.get("region_order", {})
            tasks = self.milp_params["tasks"]
            accessibility = self.milp_params["accessibility"]

            x_warm: Dict[Tuple[str, str], int] = {}
            t_warm: Dict[str, float] = {}

            for task_id in tasks:
                for arm in ["L", "R"]:
                    if arm in accessibility[task_id]:
                        x_warm[(task_id, arm)] = 0

            for a in actions:
                task_id = a["task"]
                arm = a["arm"]
                x_warm[(task_id, arm)] = 1
                t_warm[task_id] = float(a["start"])

            T_warm = float(max(a["end"] for a in actions)) if actions else 0.0

            for (task_id, arm), value in x_warm.items():
                try:
                    x[task_id, arm].Start = value
                except Exception:
                    pass
            for task_id in tasks:
                try:
                    t[task_id].Start = float(t_warm.get(task_id, 0.0))
                except Exception:
                    pass
            try:
                T.Start = T_warm
            except Exception:
                pass

            task_index = {tid: idx for idx, tid in enumerate(tasks)}

            for (i, j, arm), v in order_vars.items():
                v.Start = 1 if task_index.get(i, 0) < task_index.get(j, 0) else 0

            for arm in ["L", "R"]:
                arm_actions = sorted([a for a in actions if a["arm"] == arm],
                                     key=lambda z: z["start"])
                tasks_on_arm = [a["task"] for a in arm_actions]
                for k in range(len(tasks_on_arm)):
                    for m in range(k + 1, len(tasks_on_arm)):
                        i = tasks_on_arm[k]
                        j = tasks_on_arm[m]
                        v_ij = order_vars.get((i, j, arm))
                        v_ji = order_vars.get((j, i, arm))
                        if v_ij is not None:
                            v_ij.Start = 1
                        if v_ji is not None:
                            v_ji.Start = 0

            for (i, j), v in region_order_vars.items():
                v.Start = 1 if task_index.get(i, 0) < task_index.get(j, 0) else 0

            interference_tasks = self.milp_params["interference_set"]
            b2_tasks = [tid for tid in interference_tasks if self.milp_params["regions"][tid] == "B2"]
            b5_tasks = [tid for tid in interference_tasks if self.milp_params["regions"][tid] == "B5"]

            for region_tasks in [b2_tasks, b5_tasks]:
                task_times = [(tid, t_warm[tid]) for tid in region_tasks if tid in t_warm]
                task_times.sort(key=lambda x: x[1])
                sorted_tids = [x[0] for x in task_times]
                for k in range(len(sorted_tids)):
                    for m in range(k + 1, len(sorted_tids)):
                        i = sorted_tids[k]
                        j = sorted_tids[m]
                        v_ij = region_order_vars.get((i, j))
                        v_ji = region_order_vars.get((j, i))
                        if v_ij is not None:
                            v_ij.Start = 1
                        if v_ji is not None:
                            v_ji.Start = 0

            print(f"Warm start values set successfully. Heuristic makespan: {T_warm:.2f}s")

        except Exception as e:
            print(f"Warning: Could not set warm start values: {e}")

    # ----------------------------------------------------------
    # 保存结果（共用）
    # ----------------------------------------------------------
    def save_results(self, result_dir: str = "result") -> None:
        if self.heuristic_actions is None:
            raise ValueError("No results to save. Run solve_optimization() first.")

        os.makedirs(result_dir, exist_ok=True)

        heuristic_makespan = max(action["end"] for action in self.heuristic_actions)
        self.animate_solution(self.heuristic_actions,
                              save_path=f"{result_dir}/heuristic_animation.gif")
        self.plot_gantt_chart(self.heuristic_actions,
                              save_path=f"{result_dir}/heuristic_gantt.png",
                              title_suffix=f" - Heuristic (Makespan: {heuristic_makespan:.2f}s)")

        if self.milp_actions is not None:
            milp_makespan = max(action["end"] for action in self.milp_actions)
            self.animate_solution(self.milp_actions,
                                  save_path=f"{result_dir}/milp_animation.gif")
            self.plot_gantt_chart(self.milp_actions,
                                  save_path=f"{result_dir}/milp_gantt.png",
                                  title_suffix=f" - MILP Optimized (Makespan: {milp_makespan:.2f}s)")

            with open(f"{result_dir}/comparison_summary.txt", "w") as f:
                f.write("=== Strawberry Harvesting Optimization Results ===\n\n")
                f.write(f"Heuristic Makespan: {heuristic_makespan:.2f}s\n")
                f.write(f"MILP Makespan: {milp_makespan:.2f}s\n")
                f.write(f"Improvement: {self.improvement:.1f}%\n\n")
                f.write(f"Total Tasks: {len(self.task_df)}\n")
                if self.improvement > 0:
                    f.write(f"\nMILP achieved {self.improvement:.1f}% improvement!\n")
                else:
                    f.write(f"\nHeuristic solution was optimal or MILP could not improve it.\n")

        print(f"Results saved to {result_dir}/ directory")
        plt.close('all')

    # ----------------------------------------------------------
    # 动画（共用）
    # ----------------------------------------------------------
    def animate_solution(self, actions: List[Dict],
                         save_path: Optional[str] = None,
                         title: str = "Dual-Arm Harvesting Animation"
                         ) -> animation.FuncAnimation:
        if self.task_df is None:
            raise ValueError("No task data loaded.")

        fig, ax = plt.subplots(figsize=(8, 8))

        for i, region in enumerate(self.regions):
            cx, cy = region["center"]
            w, h = region["width"], region["height"]
            ax.add_patch(plt.Rectangle((cx - w/2, cy - h/2), w, h,
                                       fill=False, edgecolor=self.colors[i],
                                       linewidth=2, label=region["name"]))

        arm_L, = ax.plot(*self.L_base, 'ks', markersize=10, label='Left Arm Base (L)')
        arm_R, = ax.plot(*self.R_base, 'ko', markersize=10, label='Right Arm Base (R)')

        home_L, = ax.plot(*self.L_home[:2], 'b*', markersize=12, label='Left Arm Home')
        home_R, = ax.plot(*self.R_home[:2], 'r*', markersize=12, label='Right Arm Home')

        ax.add_patch(plt.Rectangle((-0.5, -0.25), 1.0, 0.5, fill=False,
                                   edgecolor='black', linestyle='--', linewidth=1.5,
                                   label='Operation Boundary'))

        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_title(title)

        task_points = {}
        task_states = {}

        for act in actions:
            task_id = act['task']
            region_name = self.task_df.iloc[int(task_id[1:])]['region']
            region_idx = next(i for i, r in enumerate(self.regions) if r['name'] == region_name)
            color = self.colors[region_idx]
            point, = ax.plot(act['x'], act['y'], 'o', color=color, markersize=8, alpha=0.8)
            task_points[task_id] = point
            task_states[task_id] = False

        line_L, = ax.plot([], [], color='blue', lw=3, alpha=0.8, label='Left Arm')
        line_R, = ax.plot([], [], color='red',  lw=3, alpha=0.8, label='Right Arm')
        time_text = ax.text(-0.95, 0.95, '', fontsize=14,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        ax.legend(loc='upper right')

        def init():
            line_L.set_data([], [])
            line_R.set_data([], [])
            time_text.set_text('')
            return [arm_L, arm_R, home_L, home_R, line_L, line_R, time_text] + list(task_points.values())

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
                    region_name = self.task_df.iloc[int(task_id[1:])]['region']
                    region_idx = next(i for i, r in enumerate(self.regions) if r['name'] == region_name)
                    new_point, = ax.plot(act['x'], act['y'], 'x',
                                        color=self.colors[region_idx],
                                        markersize=12, markeredgewidth=3)
                    task_points[task_id] = new_point
            return [arm_L, arm_R, home_L, home_R, line_L, line_R, time_text] + list(task_points.values())

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

    # ----------------------------------------------------------
    # 甘特图（共用）
    # ----------------------------------------------------------
    def plot_gantt_chart(self, actions: List[Dict],
                         save_path: Optional[str] = None,
                         title_suffix: str = "") -> None:
        if self.task_df is None:
            raise ValueError("No task data loaded.")

        fig, ax = plt.subplots(figsize=(12, 6))

        region_colors = {
            'B1': 'red', 'B2': 'orange', 'B3': 'green',
            'B4': 'blue', 'B5': 'purple', 'B6': 'brown'
        }

        left_actions  = [a for a in actions if a['arm'] == 'L']
        right_actions = [a for a in actions if a['arm'] == 'R']

        for action in left_actions:
            task_idx = int(action['task'][1:])
            region = self.task_df.iloc[task_idx]['region']
            ax.barh(0, action['end'] - action['start'], left=action['start'],
                    height=0.4, color=region_colors[region], alpha=0.7, edgecolor='black')
            mid_time = (action['start'] + action['end']) / 2
            ax.text(mid_time, 0, f"{action['task']}\n{region}",
                    ha='center', va='center', fontsize=8, fontweight='bold')

        for action in right_actions:
            task_idx = int(action['task'][1:])
            region = self.task_df.iloc[task_idx]['region']
            ax.barh(1, action['end'] - action['start'], left=action['start'],
                    height=0.4, color=region_colors[region], alpha=0.7, edgecolor='black')
            mid_time = (action['start'] + action['end']) / 2
            ax.text(mid_time, 1, f"{action['task']}\n{region}",
                    ha='center', va='center', fontsize=8, fontweight='bold')

        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Left Arm (L)', 'Right Arm (R)'])
        ax.set_xlabel('Time (seconds)')
        ax.set_title(f'Dual-Arm Task Scheduling Gantt Chart{title_suffix}')
        ax.grid(True, axis='x', alpha=0.3)

        legend_elements = [plt.Rectangle((0, 0), 1, 1,
                                         facecolor=color, alpha=0.7,
                                         edgecolor='black', label=region)
                           for region, color in region_colors.items()]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))

        makespan = max(action['end'] for action in actions) if actions else 0
        ax.axvline(x=makespan, color='red', linestyle='--', linewidth=2, alpha=0.8)
        ax.text(makespan, 0.5, f'Makespan: {makespan:.2f}s', rotation=90,
                ha='right', va='center', fontweight='bold', color='red')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gantt chart saved to {save_path}")
            plt.close(fig)
        else:
            plt.show()

    # ----------------------------------------------------------
    # 综合保存结果（共用）
    # ----------------------------------------------------------
    def save_comprehensive_results(self,
                                   result_dir: str = "comprehensive_results",
                                   experiment_name: str = None,
                                   include_data: bool = True,
                                   include_config: bool = True) -> None:
        if self.heuristic_actions is None:
            raise ValueError("No results to save. Run solve_optimization() first.")

        if experiment_name is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            experiment_name = f"dual_arm_experiment_{timestamp}"

        base_dir = os.path.join(result_dir, experiment_name)
        folders = {
            'animations': os.path.join(base_dir, 'animations'),
            'plots':      os.path.join(base_dir, 'plots'),
            'data':       os.path.join(base_dir, 'data'),
            'analysis':   os.path.join(base_dir, 'analysis'),
            'config':     os.path.join(base_dir, 'config')
        }
        for folder in folders.values():
            os.makedirs(folder, exist_ok=True)

        print(f"Saving comprehensive results to: {base_dir}")

        heuristic_makespan = max(action["end"] for action in self.heuristic_actions)
        milp_makespan = (max(action["end"] for action in self.milp_actions)
                         if self.milp_actions is not None else None)

        print("  → Generating animations...")
        self.animate_solution(
            self.heuristic_actions,
            save_path=os.path.join(folders['animations'], 'heuristic_animation.gif'),
            title=f"Heuristic Solution - Makespan: {heuristic_makespan:.2f}s")

        if self.milp_actions is not None:
            self.animate_solution(
                self.milp_actions,
                save_path=os.path.join(folders['animations'], 'milp_animation.gif'),
                title=f"MILP Solution - Makespan: {milp_makespan:.2f}s")

        print("  → Creating Gantt charts...")
        self.plot_gantt_chart(
            self.heuristic_actions,
            save_path=os.path.join(folders['plots'], 'heuristic_gantt.png'),
            title_suffix=f" - Heuristic (Makespan: {heuristic_makespan:.2f}s)")

        if self.milp_actions is not None:
            self.plot_gantt_chart(
                self.milp_actions,
                save_path=os.path.join(folders['plots'], 'milp_gantt.png'),
                title_suffix=f" - MILP Optimized (Makespan: {milp_makespan:.2f}s)")

        print("  → Creating task distribution plot...")
        self._plot_task_distribution(os.path.join(folders['plots'], 'task_distribution.png'))

        print("  → Creating region utilization analysis...")
        self._plot_region_utilization(folders['plots'])

        if include_data:
            print("  → Saving raw data...")
            self._save_raw_data(folders['data'])

        if include_config:
            print("  → Saving configuration...")
            self._save_configuration(folders['config'])

        print("  → Creating comprehensive summary...")
        self._save_comprehensive_summary(base_dir, heuristic_makespan, milp_makespan)

        if self.milp_actions is not None:
            print("  → Creating performance comparison...")
            self._create_performance_comparison(folders['analysis'])

        print(f"✓ Comprehensive results saved successfully!")
        plt.close('all')

    def _plot_task_distribution(self, save_path: str) -> None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        region_counts = self.task_df['region'].value_counts().sort_index()
        colors = [self.colors[i] for i in range(len(region_counts))]

        ax1.bar(region_counts.index, region_counts.values,
                color=colors, alpha=0.7, edgecolor='black')
        ax1.set_title('Task Count by Region')
        ax1.set_xlabel('Region')
        ax1.set_ylabel('Number of Tasks')
        ax1.grid(True, alpha=0.3)

        for i, region in enumerate(self.regions):
            region_tasks = self.task_df[self.task_df['region'] == region['name']]
            if not region_tasks.empty:
                ax2.scatter(region_tasks['x'], region_tasks['y'],
                            color=self.colors[i], label=region['name'], s=50, alpha=0.7)

        for i, region in enumerate(self.regions):
            cx, cy = region["center"]
            w, h = region["width"], region["height"]
            ax2.add_patch(plt.Rectangle((cx - w/2, cy - h/2), w, h,
                                        fill=False, edgecolor=self.colors[i], linewidth=2))

        ax2.plot(*self.L_base, 'ks', markersize=12, label='Left Arm Base')
        ax2.plot(*self.R_base, 'ko', markersize=12, label='Right Arm Base')
        ax2.plot(*self.L_home[:2], 'b*', markersize=14, label='Left Arm Home')
        ax2.plot(*self.R_home[:2], 'r*', markersize=14, label='Right Arm Home')
        ax2.set_title('Task Spatial Distribution')
        ax2.set_xlabel('X Position (m)')
        ax2.set_ylabel('Y Position (m)')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def _plot_region_utilization(self, plots_dir: str) -> None:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))

        solutions = [("Heuristic", self.heuristic_actions), ("MILP", self.milp_actions)]

        for idx, (method, actions) in enumerate(solutions):
            if actions is None:
                continue
            ax = [ax1, ax2][idx]
            region_times = {"L": {}, "R": {}}
            for action in actions:
                arm = action["arm"]
                task_idx = int(action["task"][1:])
                region = self.task_df.iloc[task_idx]["region"]
                if region not in region_times[arm]:
                    region_times[arm][region] = 0
                region_times[arm][region] += action["end"] - action["start"]

            regions = [r["name"] for r in self.regions]
            left_times  = [region_times["L"].get(region, 0) for region in regions]
            right_times = [region_times["R"].get(region, 0) for region in regions]
            x = np.arange(len(regions))
            width = 0.35

            ax.bar(x - width/2, left_times,  width, label='Left Arm',  color='blue', alpha=0.7)
            ax.bar(x + width/2, right_times, width, label='Right Arm', color='red',  alpha=0.7)
            ax.set_xlabel('Region')
            ax.set_ylabel('Total Time (s)')
            ax.set_title(f'Region Utilization - {method}')
            ax.set_xticks(x)
            ax.set_xticklabels(regions)
            ax.legend()
            ax.grid(True, alpha=0.3)

        if self.milp_actions is not None:
            methods = ["Heuristic", "MILP"]
            solutions_list = [self.heuristic_actions, self.milp_actions]

            left_workloads  = []
            right_workloads = []

            for actions in solutions_list:
                left_time  = sum(a["end"] - a["start"] for a in actions if a["arm"] == "L")
                right_time = sum(a["end"] - a["start"] for a in actions if a["arm"] == "R")
                left_workloads.append(left_time)
                right_workloads.append(right_time)

            x = np.arange(len(methods))
            width = 0.35
            ax3.bar(x - width/2, left_workloads,  width, label='Left Arm',  color='blue', alpha=0.7)
            ax3.bar(x + width/2, right_workloads, width, label='Right Arm', color='red',  alpha=0.7)
            ax3.set_xlabel('Solution Method')
            ax3.set_ylabel('Total Workload (s)')
            ax3.set_title('Arm Workload Comparison')
            ax3.set_xticks(x)
            ax3.set_xticklabels(methods)
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            for i, (left, right) in enumerate(zip(left_workloads, right_workloads)):
                ax3.text(i - width/2, left  + 0.1, f'{left:.1f}',  ha='center', va='bottom')
                ax3.text(i + width/2, right + 0.1, f'{right:.1f}', ha='center', va='bottom')

        if self.milp_actions is not None:
            self._plot_arm_timeline(ax4)

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'region_utilization.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

    def _plot_arm_timeline(self, ax) -> None:
        actions = self.milp_actions if self.milp_actions else self.heuristic_actions
        for action in [a for a in actions if a['arm'] == 'L']:
            ax.barh(0, action['end'] - action['start'], left=action['start'],
                    height=0.3, color='blue', alpha=0.7)
        for action in [a for a in actions if a['arm'] == 'R']:
            ax.barh(1, action['end'] - action['start'], left=action['start'],
                    height=0.3, color='red', alpha=0.7)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Left Arm', 'Right Arm'])
        ax.set_xlabel('Time (s)')
        ax.set_title('Arm Activity Timeline')
        ax.grid(True, alpha=0.3)

    def _save_raw_data(self, data_dir: str) -> None:
        self.task_df.to_csv(os.path.join(data_dir, 'task_data.csv'), index=False)
        if self.heuristic_actions:
            pd.DataFrame(self.heuristic_actions).to_csv(
                os.path.join(data_dir, 'heuristic_actions.csv'), index=False)
        if self.milp_actions:
            pd.DataFrame(self.milp_actions).to_csv(
                os.path.join(data_dir, 'milp_actions.csv'), index=False)

    def _save_configuration(self, config_dir: str) -> None:
        import json
        config = {
            "arm_bases": {
                "left":  self.L_base.tolist(),
                "right": self.R_base.tolist()
            },
            "arm_homes": {
                "left":  self.L_home[:2].tolist(),
                "right": self.R_home[:2].tolist()
            },
            "regions": self.regions,
            "interference_regions": self.interference_regions,
            "total_tasks": len(self.task_df) if self.task_df is not None else 0,
            "region_colors": self.colors
        }
        with open(os.path.join(config_dir, 'experiment_config.json'), 'w') as f:
            json.dump(config, f, indent=2)

    def _save_comprehensive_summary(self, base_dir: str,
                                    heuristic_makespan: float,
                                    milp_makespan: Optional[float]) -> None:
        summary_path = os.path.join(base_dir, 'EXPERIMENT_SUMMARY.md')
        with open(summary_path, 'w') as f:
            f.write("# Dual-Arm Harvesting Optimization Results\n\n")
            f.write(f"**Experiment Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Problem Configuration\n")
            f.write(f"- **Total Tasks**: {len(self.task_df)}\n")
            f.write(f"- **Regions**: {', '.join([r['name'] for r in self.regions])}\n")
            f.write(f"- **Interference Regions**: {', '.join(self.interference_regions)}\n")
            f.write(f"- **Left Arm Base**: ({self.L_base[0]:.5f}, {self.L_base[1]:.5f})\n")
            f.write(f"- **Right Arm Base**: ({self.R_base[0]:.5f}, {self.R_base[1]:.5f})\n")
            f.write(f"- **Left Arm Home**: ({self.L_home[0]:.5f}, {self.L_home[1]:.5f})\n")
            f.write(f"- **Right Arm Home**: ({self.R_home[0]:.5f}, {self.R_home[1]:.5f})\n\n")
            f.write("## Task Distribution by Region\n")
            for region, count in self.task_df['region'].value_counts().sort_index().items():
                f.write(f"- **{region}**: {count} tasks\n")
            f.write("\n## Solution Performance\n\n")
            f.write("| Method | Makespan (s) |\n")
            f.write("|--------|--------------|\n")
            f.write(f"| Heuristic | {heuristic_makespan:.2f} |\n")
            if milp_makespan is not None:
                f.write(f"| MILP | {milp_makespan:.2f} |\n")
                f.write(f"| **Improvement** | **{self.improvement:.1f}%** |\n\n")
            else:
                f.write("| MILP | No solution found |\n\n")
        print(f"  ✓ Summary saved to: EXPERIMENT_SUMMARY.md")

    def _create_performance_comparison(self, analysis_dir: str) -> None:
        if self.milp_actions is None:
            return

        fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(16, 6))

        makespans = [
            max(action["end"] for action in self.heuristic_actions),
            max(action["end"] for action in self.milp_actions)
        ]
        methods = ['Heuristic', 'MILP']

        bars1 = ax1.bar(methods, makespans,
                        color=['lightblue', 'lightgreen'], edgecolor='black', alpha=0.7)
        ax1.set_ylabel('Makespan (seconds)')
        ax1.set_title('Makespan Comparison')
        ax1.grid(True, alpha=0.3)

        for bar, value in zip(bars1, makespans):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                     f'{value:.2f}s', ha='center', va='bottom', fontweight='bold')

        improvement = ((makespans[0] - makespans[1]) / makespans[0]) * 100
        ax1.text(0.5, max(makespans) * 0.9, f'Improvement: {improvement:.1f}%',
                 ha='center', transform=ax1.transData, fontsize=12, fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

        for method, actions, color in [
            ('Heuristic', self.heuristic_actions, 'blue'),
            ('MILP',      self.milp_actions,      'green')
        ]:
            completion_times = sorted([a["end"] for a in actions])
            ax3.plot(completion_times, list(range(1, len(completion_times) + 1)),
                     marker='o', label=method, color=color, alpha=0.7, linewidth=2)

        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Tasks Completed')
        ax3.set_title('Task Completion Timeline')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, 'performance_comparison.png'),
                    dpi=300, bbox_inches='tight')
        plt.close(fig)


# ============================================================
# Baseline 子类：手工几何 + 欧氏距离近似
# ============================================================

class BaselinePlanner(DualArmPlannerCore):
    """
    Baseline 规划模式。
    - 机械臂基座：手工指定（默认 L=[-0.3, 0], R=[0.3, 0]）。
    - 单程移动代价：欧氏距离近似。
    - 任务采样：在矩形区域内连续均匀随机采样。
    - 启发式顺序：左臂 B2→B1→B4，右臂 B5→B6→B3。
    """

    def __init__(self,
                 L_base: np.ndarray = np.array([-0.3, 0.0]),
                 R_base: np.ndarray = np.array([ 0.3, 0.0]),
                 L_home: np.ndarray = np.array([-0.10, 0.20]),  # 粗略几何替代
                 R_home: np.ndarray = np.array([ 0.10, -0.20]), # 粗略几何替代
                 regions_config: Optional[List[Dict]] = None,
                 base_operation_time: float = 0.0):
        """
        参数:
            L_base: 左臂基座位置 (x, y)
            R_base: 右臂基座位置 (x, y)
            L_home: 左臂末端回篮位置 (x, y)
            R_home: 右臂末端回篮位置 (x, y)
            regions_config: 采摘区域配置；为 None 时使用默认 B1-B6。
            base_operation_time: 固定处理时间。
        """
        # 在 super().__init__() 之前设置基座，确保可视化方法可用
        self.L_base = np.asarray(L_base, dtype=float)
        self.R_base = np.asarray(R_base, dtype=float)
        self.L_home = np.asarray(L_home, dtype=float)
        self.R_home = np.asarray(R_home, dtype=float)
        super().__init__(regions_config=regions_config,
                         base_operation_time=base_operation_time)

    # ----------------------------------------------------------
    # 代价函数：欧氏距离
    # ----------------------------------------------------------
    def _compute_processing_time(self, point: np.ndarray, arm: str) -> float:
        home = self.L_home[:2] if arm == "L" else self.R_home[:2]
        one_way_time = float(np.linalg.norm(np.asarray(point)[:2] - home))
        return 2 * one_way_time + self.base_operation_time

    # ----------------------------------------------------------
    # 随机点生成
    # ----------------------------------------------------------
    def generate_random_points(self,
                                center: Tuple[float, float],
                                width: float,
                                height: float,
                                n: int = 5) -> np.ndarray:
        cx, cy = center
        xs = np.random.uniform(cx - width / 2, cx + width / 2, n)
        ys = np.random.uniform(cy - height / 2, cy + height / 2, n)
        return np.vstack((xs, ys)).T

    # ----------------------------------------------------------
    # 创建任务数据集（连续随机采样）
    # ----------------------------------------------------------
    def create_task_dataset(self, points_per_region: Dict[str, int]) -> pd.DataFrame:
        task_data = []
        for region in self.regions:
            cx, cy = region["center"]
            w, h = region["width"], region["height"]
            arm_access = region["arm_access"]
            name = region["name"]
            num_points = points_per_region.get(name, 5)
            points = self.generate_random_points((cx, cy), w, h, num_points)
            for pt in points:
                time_L = self._compute_processing_time(pt, "L") if "L" in arm_access else None
                time_R = self._compute_processing_time(pt, "R") if "R" in arm_access else None
                task_data.append({
                    "region": name,
                    "x": float(pt[0]),
                    "y": float(pt[1]),
                    "accessible_by": arm_access,
                    "time_to_L": time_L,
                    "time_to_R": time_R,
                })
        self.task_df = pd.DataFrame(task_data)
        self._extract_milp_parameters()
        return self.task_df

    # ----------------------------------------------------------
    # 加载自定义任务位置
    # ----------------------------------------------------------
    def load_task_locations(self, task_locations: List[Dict]) -> pd.DataFrame:
        task_data = []
        for task in task_locations:
            x = float(task["x"])
            y = float(task["y"])
            region_name = task["region"]
            region_config = next((r for r in self.regions if r["name"] == region_name), None)
            if region_config is None:
                raise ValueError(f"Region {region_name} not found in configuration.")
            arm_access = region_config["arm_access"]
            pt = np.array([x, y])
            time_L = self._compute_processing_time(pt, "L") if "L" in arm_access else None
            time_R = self._compute_processing_time(pt, "R") if "R" in arm_access else None
            task_data.append({
                "region": region_name,
                "x": x, "y": y,
                "accessible_by": arm_access,
                "time_to_L": time_L,
                "time_to_R": time_R,
            })
        self.task_df = pd.DataFrame(task_data)
        self._extract_milp_parameters()
        return self.task_df


# ============================================================
# RealCost 子类：URDF 几何 + cost table 查表
# ============================================================

# 默认路径（相对于 planning/ 目录的同级目录）
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_URDF = os.path.join(_REPO_ROOT, "urdf", "dual_arm_ik_xy_centered.urdf")
_DEFAULT_COST_TABLE = os.path.join(_REPO_ROOT, "roi", "results", "dual_arm_cost.pkl")


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


_DEFAULT_INTERFERENCE_LABELS = os.path.join(_REPO_ROOT, "roi", "results", "interference_labels.pkl")


class RealCostPlanner(DualArmPlannerCore):
    """
    Real Cost 规划模式。
    - 机械臂基座：从 URDF 文件（dual_arm_ik_xy_centered.urdf）解析，与 build_roi_table.py 完全一致。
    - 单程移动代价：从 cost table（dual_arm_cost.pkl）查表，不使用欧氏距离。
    - 任务采样：只从 cost table 中的可达网格离散点里采样，不支持连续随机采样。
    - 区域约束：B1/B4 强制 L-only，B3/B6 强制 R-only；B2/B5 按点实际可达性。
    - 干涉判定：不再使用 B2/B5 正方形区域，而是基于预计算的全身扫掠包络重叠危险区
      （interference_labels.pkl），以点集形式标记每个草莓目标是否为干涉目标。
    - 调度规则：分别对上半区域和下半区域，被标记为干涉的任务不允许双臂同时执行，
      但上下半区域之间的干涉任务可以并行；非干涉任务始终可以并行。
    """

    # B1/B4 强制 L-only；B3/B6 强制 R-only；B2/B5 按点可达性
    _REGION_FORCE_L = {"B1", "B4"}
    _REGION_FORCE_R = {"B3", "B6"}

    def __init__(self,
                 urdf_path: str = _DEFAULT_URDF,
                 cost_table_path: str = _DEFAULT_COST_TABLE,
                 interference_labels_path: str = _DEFAULT_INTERFERENCE_LABELS,
                 q_home: Optional[np.ndarray] = None,
                 regions_config: Optional[List[Dict]] = None,
                 base_operation_time: float = 0.0):
        """
        参数:
            urdf_path: dual_arm_ik_xy_centered.urdf 文件路径。
            cost_table_path: dual_arm_cost.pkl 文件路径。
            interference_labels_path: interference_labels.pkl 文件路径。
            q_home: 机械臂处于回篮的初始配置。未指定时将使用和 cost table 构建时一样的默认姿态。
            regions_config: 采摘区域配置；为 None 时使用默认 B1-B6。
            base_operation_time: 固定处理时间。
        """
        # ---- 1. 从 URDF 加载机械臂基座（唯一来源，不使用手工值）----
        if not os.path.isfile(urdf_path):
            raise FileNotFoundError(f"URDF not found: {urdf_path}")
        self.L_base, self.R_base = _load_urdf_base_xy(urdf_path)

        # ---- 2. 加载 cost table ----
        if not os.path.isfile(cost_table_path):
            raise FileNotFoundError(f"Cost table not found: {cost_table_path}")
        with open(cost_table_path, "rb") as f:
            _ct = pickle.load(f)
        self._left_cost:  Dict[str, float] = _ct["left_cost_table"]
        self._right_cost: Dict[str, float] = _ct["right_cost_table"]
        self._cost_table_path = cost_table_path
        
        # 尝试从 cost table 恢复 L_home/R_home
        if "left_home_xy" in _ct and "right_home_xy" in _ct and q_home is None:
            self.L_home = np.array(_ct["left_home_xy"], dtype=float)
            self.R_home = np.array(_ct["right_home_xy"], dtype=float)
        else:
            # ---- 1.5 从 URDF + q_home 动态解析 L_home/R_home 末端位置 ----
            if q_home is None:
                q_home = np.array([0.10, -1.57, 0.23, -2.20, -1.40, -2.80, 0.0], dtype=float)

            try:
                from tracikpy import TracIKSolver
                solver_l = TracIKSolver(urdf_path, "vehicle_link", "left_arm_ee_link")
                solver_r = TracIKSolver(urdf_path, "vehicle_link", "right_arm_ee_link")
                self.L_home = solver_l.fk(q_home)[:2, 3]
                self.R_home = solver_r.fk(q_home)[:2, 3]
            except ImportError:
                # 后备方案：使用 compute_danger_zone 中的 URDFArmChain FK
                import sys as _sys
                _roi_dir = os.path.join(_REPO_ROOT, "roi")
                if _roi_dir not in _sys.path:
                    _sys.path.insert(0, _roi_dir)
                from compute_danger_zone import URDFArmChain
                _chain_l = URDFArmChain(urdf_path, "vehicle_to_left_arm", "left_arm")
                _chain_r = URDFArmChain(urdf_path, "vehicle_to_right_arm", "right_arm")
                pts_l = _chain_l.fk_all_frames(q_home)
                pts_r = _chain_r.fk_all_frames(q_home)
                self.L_home = pts_l[-1][:2]  # EE frame XY
                self.R_home = pts_r[-1][:2]

        self._urdf_path = urdf_path

        # ---- 2. 加载 cost table (re-assign to avoid duplicate) ----
        if not os.path.isfile(cost_table_path):
            raise FileNotFoundError(f"Cost table not found: {cost_table_path}")
        with open(cost_table_path, "rb") as f:
            _ct = pickle.load(f)
        self._left_cost:  Dict[str, float] = _ct["left_cost_table"]
        self._right_cost: Dict[str, float] = _ct["right_cost_table"]
        self._cost_table_path = cost_table_path

        # ---- 3. 加载干涉标签 ----
        self._interference_labels: Dict[str, Dict] = {}
        if os.path.isfile(interference_labels_path):
            with open(interference_labels_path, "rb") as f:
                self._interference_labels = pickle.load(f)
            print(f"[RealCostPlanner] Interference labels: {interference_labels_path} "
                  f"({len(self._interference_labels)} entries)")
        else:
            print(f"[RealCostPlanner] WARNING: Interference labels not found: "
                  f"{interference_labels_path}")
            print("  All tasks will be treated as non-interference.")

        # ---- 4. 初始化基类 ----
        super().__init__(regions_config=regions_config,
                         base_operation_time=base_operation_time)

        print(f"[RealCostPlanner] URDF: {urdf_path}")
        print(f"[RealCostPlanner] Left  base (x,y): {self.L_base}")
        print(f"[RealCostPlanner] Right base (x,y): {self.R_base}")
        print(f"[RealCostPlanner] Cost table: {cost_table_path} "
              f"(L={len(self._left_cost)}, R={len(self._right_cost)} entries)")

    # ----------------------------------------------------------
    # 代价函数：cost table 查表
    # ----------------------------------------------------------
    def _compute_processing_time(self, point: np.ndarray, arm: str) -> Optional[float]:
        """从 cost table 查找单程代价，返回 2*cost + base_operation_time。"""
        key = f"{float(point[0]):.3f}_{float(point[1]):.3f}_0.560"
        cost = self._left_cost.get(key) if arm == "L" else self._right_cost.get(key)
        if cost is None:
            return None
        return 2.0 * cost + self.base_operation_time

    # ----------------------------------------------------------
    # 干涉判定：基于预计算的干涉标签
    # ----------------------------------------------------------
    def _is_interference_point(self, x: float, y: float,
                               accessible_by: List[str]) -> bool:
        """
        判断某个目标点是否为干涉点。
        如果任一可达臂到达该点的轨迹经过危险区，则为干涉点。
        """
        key = f"{x:.3f}_{y:.3f}_0.560"
        labels = self._interference_labels.get(key)
        if labels is None:
            return False
        if "L" in accessible_by and labels.get("left_interference", False):
            return True
        if "R" in accessible_by and labels.get("right_interference", False):
            return True
        return False

    @staticmethod
    def _point_half(y: float) -> str:
        """根据 y 坐标判断目标点所在半区域。"""
        return "upper" if y > 0 else "lower"

    # ----------------------------------------------------------
    # 辅助：获取区域内可达网格点列表
    # ----------------------------------------------------------
    def _get_region_grid_points(self,
                                 region_name: str,
                                 cx: float, cy: float,
                                 w: float, h: float
                                 ) -> List[Tuple[float, float, Optional[float], Optional[float], List[str]]]:
        """
        返回 cost table 中属于该区域的所有网格点信息。
        每条记录：(x, y, time_to_L, time_to_R, accessible_by)
        区域策略：
            B1/B4 → 仅 L 可达点，accessible_by = ["L"]
            B3/B6 → 仅 R 可达点，accessible_by = ["R"]
            B2/B5 → 按点实际可达性
        """
        xmin = cx - w / 2 - 1e-9
        xmax = cx + w / 2 + 1e-9
        ymin = cy - h / 2 - 1e-9
        ymax = cy + h / 2 + 1e-9

        if region_name in self._REGION_FORCE_L:
            # B1/B4：强制 L-only，从 L 可达表中筛选
            candidates = [
                k for k in self._left_cost
                if xmin <= float(k.split('_')[0]) <= xmax
                and ymin <= float(k.split('_')[1]) <= ymax
            ]
            result = []
            for k in candidates:
                x, y = float(k.split('_')[0]), float(k.split('_')[1])
                cost_l = self._left_cost[k]
                t_l = 2.0 * cost_l + self.base_operation_time
                result.append((x, y, t_l, None, ["L"]))
            return result

        elif region_name in self._REGION_FORCE_R:
            # B3/B6：强制 R-only，从 R 可达表中筛选
            candidates = [
                k for k in self._right_cost
                if xmin <= float(k.split('_')[0]) <= xmax
                and ymin <= float(k.split('_')[1]) <= ymax
            ]
            result = []
            for k in candidates:
                x, y = float(k.split('_')[0]), float(k.split('_')[1])
                cost_r = self._right_cost[k]
                t_r = 2.0 * cost_r + self.base_operation_time
                result.append((x, y, None, t_r, ["R"]))
            return result

        else:
            # B2/B5：按点实际可达性
            l_keys = {
                k for k in self._left_cost
                if xmin <= float(k.split('_')[0]) <= xmax
                and ymin <= float(k.split('_')[1]) <= ymax
            }
            r_keys = {
                k for k in self._right_cost
                if xmin <= float(k.split('_')[0]) <= xmax
                and ymin <= float(k.split('_')[1]) <= ymax
            }
            all_keys = l_keys | r_keys
            result = []
            for k in all_keys:
                x, y = float(k.split('_')[0]), float(k.split('_')[1])
                t_l = None
                t_r = None
                accessible_by = []
                if k in l_keys:
                    t_l = 2.0 * self._left_cost[k] + self.base_operation_time
                    accessible_by.append("L")
                if k in r_keys:
                    t_r = 2.0 * self._right_cost[k] + self.base_operation_time
                    accessible_by.append("R")
                result.append((x, y, t_l, t_r, accessible_by))
            return result

    # ----------------------------------------------------------
    # 创建任务数据集（只从 cost table 网格点采样）
    # ----------------------------------------------------------
    def create_task_dataset(self, points_per_region: Dict[str, int]) -> pd.DataFrame:
        """
        在每个区域内从 cost table 可达网格点中随机采样指定数量的任务点。
        若可达点数量不足，则取全部可达点。
        """
        task_data = []
        for region in self.regions:
            cx, cy = region["center"]
            w, h = region["width"], region["height"]
            name = region["name"]
            num_points = points_per_region.get(name, 5)

            candidates = self._get_region_grid_points(name, cx, cy, w, h)

            if len(candidates) == 0:
                print(f"[RealCostPlanner] WARNING: No reachable grid points in region {name}.")
                continue

            # 随机采样（不重复）
            n_sample = min(num_points, len(candidates))
            indices = np.random.choice(len(candidates), n_sample, replace=False)

            for idx in indices:
                x, y, t_l, t_r, accessible_by = candidates[idx]
                is_interf = self._is_interference_point(x, y, accessible_by)
                task_data.append({
                    "region": name,
                    "x": x, "y": y,
                    "accessible_by": list(accessible_by),
                    "time_to_L": t_l,
                    "time_to_R": t_r,
                    "is_interference": is_interf,
                    "half": self._point_half(y),
                })

        self.task_df = pd.DataFrame(task_data)
        self._extract_milp_parameters()
        return self.task_df

    # ----------------------------------------------------------
    # 加载自定义任务位置（要求点在 cost table 中）
    # ----------------------------------------------------------
    def load_task_locations(self, task_locations: List[Dict]) -> pd.DataFrame:
        """
        从字典列表加载任务位置。坐标直接格式化为 3 位小数以匹配 cost table 键格式
        （cost table 网格步长 0.02m，键以 3 位小数记录）。
        若坐标在 cost table 中不可达，则跳过并警告。
        """
        task_data = []
        for task in task_locations:
            region_name = task["region"]
            # 直接用 3 位小数格式化，与 cost table 键格式一致
            x = float(task["x"])
            y = float(task["y"])
            key = f"{x:.3f}_{y:.3f}_0.560"

            region_config = next((r for r in self.regions if r["name"] == region_name), None)
            if region_config is None:
                raise ValueError(f"Region {region_name} not found in configuration.")

            if region_name in self._REGION_FORCE_L:
                t_l = (2.0 * self._left_cost[key] + self.base_operation_time
                       if key in self._left_cost else None)
                if t_l is None:
                    print(f"[RealCostPlanner] WARNING: ({x:.3f}, {y:.3f}) not L-reachable; skipping.")
                    continue
                accessible_by = ["L"]
                t_r = None
            elif region_name in self._REGION_FORCE_R:
                t_r = (2.0 * self._right_cost[key] + self.base_operation_time
                       if key in self._right_cost else None)
                if t_r is None:
                    print(f"[RealCostPlanner] WARNING: ({x:.3f}, {y:.3f}) not R-reachable; skipping.")
                    continue
                accessible_by = ["R"]
                t_l = None
            else:
                # B2 / B5：按点实际可达性
                has_l = key in self._left_cost
                has_r = key in self._right_cost
                if not has_l and not has_r:
                    print(f"[RealCostPlanner] WARNING: ({x:.3f}, {y:.3f}) not reachable by either arm; skipping.")
                    continue
                accessible_by = (["L"] if has_l else []) + (["R"] if has_r else [])
                t_l = (2.0 * self._left_cost[key]  + self.base_operation_time) if has_l else None
                t_r = (2.0 * self._right_cost[key] + self.base_operation_time) if has_r else None

            task_data.append({
                "region": region_name,
                "x": x, "y": y,
                "accessible_by": accessible_by,
                "time_to_L": t_l,
                "time_to_R": t_r,
                "is_interference": self._is_interference_point(x, y, accessible_by),
                "half": self._point_half(y),
            })

        self.task_df = pd.DataFrame(task_data)
        self._extract_milp_parameters()
        return self.task_df

    # ----------------------------------------------------------
    # 参数提取：覆写基类以使用干涉标签
    # ----------------------------------------------------------
    def _extract_milp_parameters(self):
        """
        覆写基类的参数提取方法，使用 is_interference + half 替代 B2/B5 区域判定。
        interference_set 按上下半区域分为两组。
        """
        parameters = {
            "tasks": [],
            "positions": {},
            "regions": {},
            "accessibility": {},
            "processing_time": {},
            "interference_set": set(),
            "upper_interference": [],   # 上半区域干涉任务
            "lower_interference": [],   # 下半区域干涉任务
        }
        for idx, row in self.task_df.iterrows():
            task_id = f"t{idx}"
            parameters["tasks"].append(task_id)
            parameters["positions"][task_id] = (row["x"], row["y"])
            parameters["regions"][task_id] = row["region"]
            parameters["accessibility"][task_id] = row["accessible_by"]
            if row.get("is_interference", False):
                parameters["interference_set"].add(task_id)
                half = row.get("half", self._point_half(row["y"]))
                if half == "upper":
                    parameters["upper_interference"].append(task_id)
                else:
                    parameters["lower_interference"].append(task_id)
            if "L" in row["accessible_by"]:
                parameters["processing_time"][(task_id, "L")] = row["time_to_L"]
            if "R" in row["accessible_by"]:
                parameters["processing_time"][(task_id, "R")] = row["time_to_R"]
        self.milp_params = parameters

    # ----------------------------------------------------------
    # MILP 模型构建：覆写以使用干涉标签
    # ----------------------------------------------------------
    def build_milp_model(self, warm_start_actions: Optional[List[Dict]] = None) -> gp.Model:
        """
        覆写基类的 MILP 构建。
        约束 4 不再按 B2/B5 区域，而是按上半区域干涉任务组和下半区域干涉任务组
        分别施加串行约束。
        """
        if self.milp_params is None:
            raise ValueError("No task data loaded.")

        tasks = self.milp_params["tasks"]
        arms = ["L", "R"]
        processing_time = self.milp_params["processing_time"]
        accessibility = self.milp_params["accessibility"]
        upper_interference = self.milp_params["upper_interference"]
        lower_interference = self.milp_params["lower_interference"]

        model = gp.Model("DualArmHarvesting")
        model.setParam('OutputFlag', 0)

        x = model.addVars(tasks, arms, vtype=GRB.BINARY, name="x")
        t = model.addVars(tasks, vtype=GRB.CONTINUOUS, name="t")
        T = model.addVar(vtype=GRB.CONTINUOUS, name="T")

        order_vars: Dict[Tuple[str, str, str], gp.Var] = {}
        region_order_vars: Dict[Tuple[str, str], gp.Var] = {}

        # 约束 1：每个任务只分配给一只手臂
        for i in tasks:
            model.addConstr(gp.quicksum(x[i, a] for a in accessibility[i]) == 1,
                            name=f"assign_{i}")

        # 约束 2：遵守可达性
        for i in tasks:
            for a in arms:
                if a not in accessibility[i]:
                    model.addConstr(x[i, a] == 0, name=f"reach_{i}_{a}")

        # 约束 3：同一手臂的顺序约束
        for idx_i in range(len(tasks)):
            for idx_j in range(idx_i + 1, len(tasks)):
                i = tasks[idx_i]
                j = tasks[idx_j]
                for a in arms:
                    if a in accessibility[i] and a in accessibility[j]:
                        o_ij = model.addVar(vtype=GRB.BINARY, name=f"order_{i}_{j}_{a}")
                        o_ji = model.addVar(vtype=GRB.BINARY, name=f"order_{j}_{i}_{a}")
                        order_vars[(i, j, a)] = o_ij
                        order_vars[(j, i, a)] = o_ji
                        model.addConstr(o_ij + o_ji == 1, name=f"order_sum_{i}_{j}_{a}")
                        M = 1000
                        model.addConstr(
                            t[i] + processing_time[i, a]
                            <= t[j] + M * (1 - o_ij) + M * (1 - x[i, a]) + M * (1 - x[j, a]),
                            name=f"seq_{i}_{j}_{a}")
                        model.addConstr(
                            t[j] + processing_time[j, a]
                            <= t[i] + M * (1 - o_ji) + M * (1 - x[i, a]) + M * (1 - x[j, a]),
                            name=f"seq_{j}_{i}_{a}")

        # 约束 4：干涉任务串行约束（按上下半区域分组）
        for group_tasks in [upper_interference, lower_interference]:
            for idx_i in range(len(group_tasks)):
                for idx_j in range(idx_i + 1, len(group_tasks)):
                    i = group_tasks[idx_i]
                    j = group_tasks[idx_j]
                    y_ij = model.addVar(vtype=GRB.BINARY, name=f"region_order_{i}_{j}")
                    y_ji = model.addVar(vtype=GRB.BINARY, name=f"region_order_{j}_{i}")
                    region_order_vars[(i, j)] = y_ij
                    region_order_vars[(j, i)] = y_ji
                    model.addConstr(y_ij + y_ji == 1, name=f"region_order_sum_{i}_{j}")
                    M = 1000
                    for a1 in accessibility[i]:
                        for a2 in accessibility[j]:
                            model.addConstr(
                                t[i] + processing_time[i, a1]
                                <= t[j] + M * (1 - y_ij) + M * (1 - x[i, a1]) + M * (1 - x[j, a2]),
                                name=f"region_seq_{i}_{j}_{a1}_{a2}")
                    for a1 in accessibility[j]:
                        for a2 in accessibility[i]:
                            model.addConstr(
                                t[j] + processing_time[j, a1]
                                <= t[i] + M * (1 - y_ji) + M * (1 - x[j, a1]) + M * (1 - x[i, a2]),
                                name=f"region_seq_{j}_{i}_{a1}_{a2}")

        # 约束 5：总完工时间下界
        for i in tasks:
            for a in accessibility[i]:
                model.addConstr(
                    t[i] + processing_time[i, a] <= T + 1000 * (1 - x[i, a]),
                    name=f"makespan_{i}_{a}")

        # 目标：最小化总完工时间
        model.setObjective(T, GRB.MINIMIZE)

        model._varmap = {
            "x": x, "t": t, "T": T,
            "order": order_vars,
            "region_order": region_order_vars,
            "tasks": tasks, "arms": arms,
        }

        if warm_start_actions is not None:
            self._set_warm_start(model, warm_start_actions)

        return model

    # ----------------------------------------------------------
    # 热启动：覆写以使用上下半区域干涉分组
    # ----------------------------------------------------------
    def _set_warm_start(self, model: gp.Model, actions: List[Dict]):
        try:
            if not hasattr(model, "_varmap") or model._varmap is None:
                raise RuntimeError("Warm start requires model._varmap.")
            varmap = model._varmap
            x = varmap["x"]
            t_var = varmap["t"]
            T = varmap["T"]
            order_vars = varmap.get("order", {})
            region_order_vars = varmap.get("region_order", {})
            tasks = self.milp_params["tasks"]
            accessibility = self.milp_params["accessibility"]

            x_warm: Dict[Tuple[str, str], int] = {}
            t_warm: Dict[str, float] = {}

            for task_id in tasks:
                for arm in ["L", "R"]:
                    if arm in accessibility[task_id]:
                        x_warm[(task_id, arm)] = 0

            for a in actions:
                task_id = a["task"]
                arm = a["arm"]
                x_warm[(task_id, arm)] = 1
                t_warm[task_id] = float(a["start"])

            T_warm = float(max(a["end"] for a in actions)) if actions else 0.0

            for (task_id, arm), value in x_warm.items():
                try:
                    x[task_id, arm].Start = value
                except Exception:
                    pass
            for task_id in tasks:
                try:
                    t_var[task_id].Start = float(t_warm.get(task_id, 0.0))
                except Exception:
                    pass
            try:
                T.Start = T_warm
            except Exception:
                pass

            task_index = {tid: idx for idx, tid in enumerate(tasks)}

            for (i, j, arm), v in order_vars.items():
                v.Start = 1 if task_index.get(i, 0) < task_index.get(j, 0) else 0

            for arm in ["L", "R"]:
                arm_actions = sorted([a for a in actions if a["arm"] == arm],
                                     key=lambda z: z["start"])
                tasks_on_arm = [a["task"] for a in arm_actions]
                for k in range(len(tasks_on_arm)):
                    for m in range(k + 1, len(tasks_on_arm)):
                        i = tasks_on_arm[k]
                        j = tasks_on_arm[m]
                        v_ij = order_vars.get((i, j, arm))
                        v_ji = order_vars.get((j, i, arm))
                        if v_ij is not None:
                            v_ij.Start = 1
                        if v_ji is not None:
                            v_ji.Start = 0

            for (i, j), v in region_order_vars.items():
                v.Start = 1 if task_index.get(i, 0) < task_index.get(j, 0) else 0

            # 按上下半区域干涉分组设置热启动
            upper_interference = self.milp_params["upper_interference"]
            lower_interference = self.milp_params["lower_interference"]
            for group_tasks in [upper_interference, lower_interference]:
                task_times = [(tid, t_warm[tid]) for tid in group_tasks if tid in t_warm]
                task_times.sort(key=lambda xx: xx[1])
                sorted_tids = [xx[0] for xx in task_times]
                for k in range(len(sorted_tids)):
                    for m in range(k + 1, len(sorted_tids)):
                        i = sorted_tids[k]
                        j = sorted_tids[m]
                        v_ij = region_order_vars.get((i, j))
                        v_ji = region_order_vars.get((j, i))
                        if v_ij is not None:
                            v_ij.Start = 1
                        if v_ji is not None:
                            v_ji.Start = 0

            print(f"Warm start values set successfully. Heuristic makespan: {T_warm:.2f}s")

        except Exception as e:
            print(f"Warning: Could not set warm start values: {e}")

    # ----------------------------------------------------------
    # 启发式：基于干涉标签的分离调度
    # ----------------------------------------------------------
    def spatial_order_heuristic(self) -> List[Dict]:
        """
        Real Cost 启发式（基于干涉标签）。

        步骤一：预分配双臂可达任务的归属
          - B1/B4 区域任务强制归 L
          - B3/B6 区域任务强制归 R
          - B2/B5 区域中：only-L → L；only-R → R；
            B2 的 both 可达任务优先归 L，B5 的优先归 R

        步骤二：分两大类调度
          - 非干涉任务（is_interference=False）：两臂可随意并行
          - 干涉任务（is_interference=True）：同一半区域内的干涉任务必须串行
            （同一半区域里，一个臂执行干涉任务时另一个臂只能执行该半区域的非干涉任务或其他半区域的任务）

        调度顺序：
          - 左臂先执行非干涉任务，再执行干涉任务
          - 右臂先执行非干涉任务，再执行干涉任务
          交替调度上下半区域干涉任务以保证互斥。
        """
        if self.task_df is None:
            raise ValueError("No task data loaded.")

        actions = []
        arm_times = {'L': 0.0, 'R': 0.0}
        # 按上下半区域跟踪干涉任务的忙碌截止时间
        interference_busy_until = {'upper': 0.0, 'lower': 0.0}

        # ---- 步骤一：预分配 ----
        l_tasks = []  # (df_idx, arm)
        r_tasks = []

        for idx, row in self.task_df.iterrows():
            ab = row['accessible_by']
            region = row['region']
            if isinstance(ab, list):
                if "L" in ab and "R" not in ab:
                    l_tasks.append(idx)
                elif "R" in ab and "L" not in ab:
                    r_tasks.append(idx)
                else:
                    # both 可达
                    if region in ("B2", "B1", "B4"):
                        l_tasks.append(idx)
                    else:
                        r_tasks.append(idx)

        # ---- 步骤二：对每个臂，先调度非干涉任务，再调度干涉任务 ----
        def _schedule_arm_tasks(arm, task_indices):
            nonlocal arm_times
            arm_actions = []
            non_interf = [i for i in task_indices
                          if not self.task_df.iloc[i].get('is_interference', False)]
            interf = [i for i in task_indices
                      if self.task_df.iloc[i].get('is_interference', False)]

            # 非干涉任务：直接顺序执行，不需要等待
            non_interf_df = self.task_df.iloc[non_interf].copy() if non_interf else pd.DataFrame()
            if not non_interf_df.empty:
                non_interf_df['proc_time'] = non_interf_df[f'time_to_{arm}']
                non_interf_df = non_interf_df.sort_values('proc_time')
                for i, row in non_interf_df.iterrows():
                    start = arm_times[arm]
                    end = start + row[f'time_to_{arm}']
                    arm_actions.append({
                        'task': f't{i}', 'arm': arm,
                        'x': row['x'], 'y': row['y'],
                        'start': start, 'end': end,
                    })
                    arm_times[arm] = end

            # 干涉任务：按半区域分组串行
            interf_df = self.task_df.iloc[interf].copy() if interf else pd.DataFrame()
            if not interf_df.empty:
                interf_df['proc_time'] = interf_df[f'time_to_{arm}']
                interf_df = interf_df.sort_values('proc_time')
                for i, row in interf_df.iterrows():
                    half = row.get('half', self._point_half(row['y']))
                    # 必须等待同半区域干涉任务完成
                    actual_start = max(arm_times[arm],
                                       interference_busy_until.get(half, 0.0))
                    end = actual_start + row[f'time_to_{arm}']
                    interference_busy_until[half] = end
                    arm_actions.append({
                        'task': f't{i}', 'arm': arm,
                        'x': row['x'], 'y': row['y'],
                        'start': actual_start, 'end': end,
                    })
                    arm_times[arm] = end

            return arm_actions

        print("Scheduling with interference labels ...")
        n_interf = sum(1 for _, r in self.task_df.iterrows()
                       if r.get('is_interference', False))
        print(f"  Interference tasks: {n_interf}, "
              f"Non-interference: {len(self.task_df) - n_interf}")

        # 交替调度：左臂一批、右臂一批
        l_actions = _schedule_arm_tasks('L', l_tasks)
        r_actions = _schedule_arm_tasks('R', r_tasks)
        actions.extend(l_actions)
        actions.extend(r_actions)

        return sorted(actions, key=lambda x: x["start"])

    # ----------------------------------------------------------
    # Gantt 图：覆写基类以显示干涉状态
    # ----------------------------------------------------------
    def plot_gantt_chart(self, actions: List[Dict],
                         save_path: Optional[str] = None,
                         title_suffix: str = "") -> None:
        if self.task_df is None:
            raise ValueError("No task data loaded.")

        fig, ax = plt.subplots(figsize=(12, 6))

        left_actions  = [a for a in actions if a['arm'] == 'L']
        right_actions = [a for a in actions if a['arm'] == 'R']

        def _bar_color(task_id):
            task_idx = int(task_id[1:])
            row = self.task_df.iloc[task_idx]
            if row.get('is_interference', False):
                return '#d65b5b'  # red for interference
            return '#6dbd7d'      # green for non-interference

        def _bar_label(task_id):
            task_idx = int(task_id[1:])
            row = self.task_df.iloc[task_idx]
            status = "S" if row.get('is_interference', False) else "P"
            half = row.get('half', 'upper' if row['y'] > 0 else 'lower')
            half_tag = "U" if half == "upper" else "L"
            return f"{task_id}\n{status}/{half_tag}"

        for action in left_actions:
            color = _bar_color(action['task'])
            ax.barh(0, action['end'] - action['start'], left=action['start'],
                    height=0.4, color=color, alpha=0.7, edgecolor='black')
            mid_time = (action['start'] + action['end']) / 2
            ax.text(mid_time, 0, _bar_label(action['task']),
                    ha='center', va='center', fontsize=7, fontweight='bold')

        for action in right_actions:
            color = _bar_color(action['task'])
            ax.barh(1, action['end'] - action['start'], left=action['start'],
                    height=0.4, color=color, alpha=0.7, edgecolor='black')
            mid_time = (action['start'] + action['end']) / 2
            ax.text(mid_time, 1, _bar_label(action['task']),
                    ha='center', va='center', fontsize=7, fontweight='bold')

        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Left Arm (L)', 'Right Arm (R)'])
        ax.set_xlabel('Time (seconds)')
        ax.set_title(f'Dual-Arm Task Scheduling Gantt Chart{title_suffix}')
        ax.grid(True, axis='x', alpha=0.3)

        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor='#d65b5b', alpha=0.7,
                           edgecolor='black', label='Serial (interference)'),
            plt.Rectangle((0, 0), 1, 1, facecolor='#6dbd7d', alpha=0.7,
                           edgecolor='black', label='Parallel (non-interference)'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))

        makespan = max(action['end'] for action in actions) if actions else 0
        ax.axvline(x=makespan, color='red', linestyle='--', linewidth=2, alpha=0.8)
        ax.text(makespan, 0.5, f'Makespan: {makespan:.2f}s', rotation=90,
                ha='right', va='center', fontweight='bold', color='red')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gantt chart saved to {save_path}")
            plt.close(fig)
        else:
            plt.show()

    # ----------------------------------------------------------
    # 动画：覆写基类以显示干涉状态
    # ----------------------------------------------------------
    def animate_solution(self, actions: List[Dict],
                         save_path: Optional[str] = None,
                         title: str = "Dual-Arm Harvesting Animation"
                         ) -> animation.FuncAnimation:
        if self.task_df is None:
            raise ValueError("No task data loaded.")

        fig, ax = plt.subplots(figsize=(8, 8))

        # 绘制区域框
        for i, region in enumerate(self.regions):
            cx, cy = region["center"]
            w, h = region["width"], region["height"]
            ax.add_patch(plt.Rectangle((cx - w/2, cy - h/2), w, h,
                                       fill=False, edgecolor=self.colors[i],
                                       linewidth=2, label=region["name"]))

        arm_L, = ax.plot(*self.L_base, 'ks', markersize=10, label='Left Arm Base (L)')
        arm_R, = ax.plot(*self.R_base, 'ko', markersize=10, label='Right Arm Base (R)')

        home_L, = ax.plot(*self.L_home[:2], 'b*', markersize=12, label='Left Arm Home')
        home_R, = ax.plot(*self.R_home[:2], 'r*', markersize=12, label='Right Arm Home')

        ax.add_patch(plt.Rectangle((-0.5, -0.25), 1.0, 0.5, fill=False,
                                   edgecolor='black', linestyle='--', linewidth=1.5,
                                   label='Operation Boundary'))

        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_title(title)

        task_points = {}
        task_states = {}

        for act in actions:
            task_id = act['task']
            task_idx = int(task_id[1:])
            row = self.task_df.iloc[task_idx]
            is_interf = row.get('is_interference', False)
            color = '#d65b5b' if is_interf else '#6dbd7d'
            point, = ax.plot(act['x'], act['y'], 'o', color=color, markersize=8, alpha=0.8)
            task_points[task_id] = (point, color)
            task_states[task_id] = False

        line_L, = ax.plot([], [], color='blue', lw=3, alpha=0.8, label='Left Arm')
        line_R, = ax.plot([], [], color='red',  lw=3, alpha=0.8, label='Right Arm')
        time_text = ax.text(-0.95, 0.95, '', fontsize=14,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        ax.legend(loc='upper right', fontsize=7)

        def init():
            line_L.set_data([], [])
            line_R.set_data([], [])
            time_text.set_text('')
            return [arm_L, arm_R, home_L, home_R, line_L, line_R, time_text] + \
                   [p for p, c in task_points.values()]

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
                    old_point, color = task_points[task_id]
                    old_point.remove()
                    new_point, = ax.plot(act['x'], act['y'], 'x',
                                        color=color,
                                        markersize=12, markeredgewidth=3)
                    task_points[task_id] = (new_point, color)
            return [arm_L, arm_R, home_L, home_R, line_L, line_R, time_text] + \
                   [p for p, c in task_points.values()]

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


# ============================================================
# 向后兼容别名
# ============================================================
DualArmPlanner = BaselinePlanner
