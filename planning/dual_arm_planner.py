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
import json
from typing import List, Dict, Tuple, Optional, Union, Any


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
                           time_limit: int = 1500,
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
                    region_name = self.task_df.iloc[int(task_id[1:])]['region']
                    region_idx = next(i for i, r in enumerate(self.regions) if r['name'] == region_name)
                    new_point, = ax.plot(act['x'], act['y'], 'x',
                                        color=self.colors[region_idx],
                                        markersize=12, markeredgewidth=3)
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
            f.write(f"- **Right Arm Base**: ({self.R_base[0]:.5f}, {self.R_base[1]:.5f})\n\n")
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
                 regions_config: Optional[List[Dict]] = None,
                 base_operation_time: float = 0.0):
        """
        参数:
            L_base: 左臂基座位置 (x, y)
            R_base: 右臂基座位置 (x, y)
            regions_config: 采摘区域配置；为 None 时使用默认 B1-B6。
            base_operation_time: 固定处理时间。
        """
        # 在 super().__init__() 之前设置基座，确保可视化方法可用
        self.L_base = np.asarray(L_base, dtype=float)
        self.R_base = np.asarray(R_base, dtype=float)
        super().__init__(regions_config=regions_config,
                         base_operation_time=base_operation_time)

    # ----------------------------------------------------------
    # 代价函数：欧氏距离
    # ----------------------------------------------------------
    def _compute_processing_time(self, point: np.ndarray, arm: str) -> float:
        base = self.L_base if arm == "L" else self.R_base
        one_way_time = float(np.linalg.norm(np.asarray(point) - base))
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




_DEFAULT_POINT_TABLE = os.path.join(_REPO_ROOT, "ompl", "results", "point_table", "point_table.pkl")


def _create_key(x: float, y: float, z: float = 0.56) -> str:
    return f"{float(x):.3f}_{float(y):.3f}_{float(z):.3f}"


def _best_candidate(cands: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not cands:
        return None
    valid = [c for c in cands if c.get("best_cost") is not None]
    if not valid:
        return None
    return min(valid, key=lambda c: float(c["best_cost"]))


class RealCostPlanner(DualArmPlannerCore):
    """
    Point-motion 版 Real Cost 规划模式。
    - 机械臂基座：从 URDF 解析（与旧版一致）
    - 数据来源：ompl/results/point_table/point_table.pkl
    - 不再使用 B1-B6 / B2-B5 干涉区逻辑
    - 直接使用 point-level motion table 中的:
        label            -> parallel / serial_upper / serial_lower
        allowed_arms     -> 允许分配的臂
        must_assign_to   -> 必须分配的臂（若有）
        optimizer_candidates_L / R
        optimizer_best_cost_L / R
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
                 regions_config: Optional[List[Dict]] = None,
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

        super().__init__(regions_config=regions_config, base_operation_time=base_operation_time)

        # Point-motion 模式不再使用旧 B2/B5 干涉区
        self.interference_regions = []

        print(f"[RealCostPlanner:point-motion] point table: {point_motion_table_path}")
        print(f"[RealCostPlanner:point-motion] URDF: {urdf_path}")
        print(f"[RealCostPlanner:point-motion] Left  base: {self.L_base}")
        print(f"[RealCostPlanner:point-motion] Right base: {self.R_base}")
        print(f"[RealCostPlanner:point-motion] total points in table: {len(self.point_index)}")

    # ----------------------------------------------------------
    # 旧接口兼容：这里只从 point_motion_table 构建数据，不再按区域随机点
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
    # 参数提取：不再使用 regions/interference_set，而使用 point labels
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
    # 新启发式：phased heuristic
    # ----------------------------------------------------------
    def spatial_order_heuristic(self) -> List[Dict]:
        """
        用户确认的 phased heuristic：

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
    # MILP：不再使用 B2/B5，而使用 serial_upper / serial_lower 串行点集
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
    # Point-motion 专用 warm start（修复旧版接口不兼容问题）
    # ----------------------------------------------------------
    def _set_warm_start(self, model: gp.Model, actions: List[Dict]):
        """
        为 point-motion MILP 设置 warm start。
        修复点：
        - 不再依赖旧版的 interference_set / regions
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
    # Point-motion 专用求解：MILP 若不优于 heuristic，则回退为 heuristic
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
    # 解提取：附加 point-motion 字段
    # ----------------------------------------------------------
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

    # ----------------------------------------------------------
    # 可视化：不再绘制 B1-B6，改为点标签视图
    # ----------------------------------------------------------
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


# ============================================================
# 向后兼容别名
# ============================================================
DualArmPlanner = BaselinePlanner
PointMotionRealCostPlanner = RealCostPlanner
