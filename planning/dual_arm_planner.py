import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import os
import time
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Union


# ===========================================================================
# 核心规划器（Core Planner）
# 包含：MILP 建模、求解、结果提取、结果保存、通用绘图骨架
# 不包含：区域配置、处理时间计算、任务生成、启发式具体实现、配色方案
# ===========================================================================

class DualArmPlannerCore:
    """
    双臂规划器核心基类。

    包含与具体任务/环境模式无关的通用逻辑：
      - MILP 建模与求解
      - 结果提取与保存
      - 动画与甘特图绘图骨架（通过钩子方法适配不同模式）

    干涉区判断依赖 task_df 中每行的 ``is_interference`` 布尔字段，
    不再使用硬编码的区域名列表。

    子类需覆盖：
      - ``spatial_order_heuristic()``：启发式算法
      - ``_draw_environment_background(ax, with_region_labels)``：环境背景绘图
      - ``_get_task_color(task_id)``：任务配色规则
    """

    def __init__(self,
                 L_base: np.ndarray = np.array([-0.3, 0]),
                 R_base: np.ndarray = np.array([0.3, 0]),
                 base_operation_time: float = 0.0):
        """
        参数:
            L_base: 左臂基座位置 (x, y)
            R_base: 右臂基座位置 (x, y)
            base_operation_time: 单个草莓的固定处理时间（包括采摘、放置等）
        """
        self.L_base = L_base
        self.R_base = R_base
        self.base_operation_time = base_operation_time

        # 数据存储
        self.task_df = None
        self.milp_params = None
        self.heuristic_actions = None
        self.milp_actions = None
        self.improvement = 0

    # ----------------------------------------------------------------------- #
    # 绘图钩子（子类覆盖以实现模式相关的可视化）                               #
    # ----------------------------------------------------------------------- #

    def _draw_environment_background(self, ax, with_region_labels: bool = True) -> None:
        """
        绘制环境背景（区域边界、操作边界等）。

        参数:
            ax: matplotlib Axes 对象
            with_region_labels: 是否为区域添加图例标签（在任务分布图中设为 False
                以避免与散点标签重复；在动画中设为 True 以显示区域图例）。
        """
        pass  # 子类覆盖

    def _get_task_color(self, task_id: str) -> str:
        """返回给定任务的绘图颜色。子类覆盖以实现模式相关的配色规则。"""
        return 'blue'  # 默认颜色

    # ----------------------------------------------------------------------- #
    # 核心方法：参数提取                                                       #
    # ----------------------------------------------------------------------- #

    def _extract_milp_parameters(self):
        """
        从 task_df 中提取 MILP 所需参数。
        干涉区判断使用 task_df 的 ``is_interference`` 字段，不依赖区域名硬编码。
        """
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
            if row["is_interference"]:
                parameters["interference_set"].add(task_id)
            if "L" in row["accessible_by"]:
                parameters["processing_time"][(task_id, "L")] = row["time_to_L"]
            if "R" in row["accessible_by"]:
                parameters["processing_time"][(task_id, "R")] = row["time_to_R"]
        self.milp_params = parameters

    # ----------------------------------------------------------------------- #
    # 核心方法：求解                                                           #
    # ----------------------------------------------------------------------- #

    def solve_optimization(self, time_limit: int = 30,
                           heuristic_name: str = "spatial_order") -> Tuple[List[Dict], List[Dict], float]:
        """
        求解目标函数的优化问题。

        参数:
            time_limit: MILP 求解器时间限制（秒）
            heuristic_name: 要使用的启发式算法名称
        """
        if self.task_df is None:
            raise ValueError("No task data loaded.")

        # 第一步：运行启发式
        print(f"Running {heuristic_name} heuristic...")
        if heuristic_name == "spatial_order":
            self.heuristic_actions = self.spatial_order_heuristic()
        else:
            raise ValueError(f"Unknown heuristic: {heuristic_name}")

        heuristic_makespan = max(action["end"] for action in self.heuristic_actions)
        print(f"Heuristic - Makespan: {heuristic_makespan:.2f}s")

        # 第二步：构建并求解 MILP
        print("Building MILP model...")
        model = self.build_milp_model(self.heuristic_actions)

        model.setParam("TimeLimit", time_limit)
        model.setParam("MIPFocus", 1)
        model.setParam("OutputFlag", 1)

        print(f"Solving MILP (time limit: {time_limit}s)...")
        model.optimize()

        # 第三步：提取结果
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

    # ----------------------------------------------------------------------- #
    # 核心方法：MILP 建模                                                      #
    # ----------------------------------------------------------------------- #

    def build_milp_model(self, warm_start_actions: Optional[List[Dict]] = None) -> gp.Model:
        """
        把 MILP 约束翻译成 Gurobi 能理解的形式。

        干涉区约束按区域动态分组，不再硬编码区域名（如 B2/B5）。

        输入:
            warm_start_actions: 用于热启动的可选动作列表
        输出:
            返回一个配置好的 gurobipy.Model 对象：
                包含了所有的决策变量、线性约束和优化目标，可以直接调用 model.optimize() 进行求解。
        """
        if self.milp_params is None:
            raise ValueError("No task data loaded. Call create_task_dataset() or load_task_locations() first.")

        tasks = self.milp_params["tasks"]
        arms = ["L", "R"]
        interference_tasks = self.milp_params["interference_set"]
        processing_time = self.milp_params["processing_time"]
        accessibility = self.milp_params["accessibility"]

        model = gp.Model("DualArmHarvesting")
        model.setParam('OutputFlag', 0)

        # 决策变量
        x = model.addVars(tasks, arms, vtype=GRB.BINARY, name="x")
        t = model.addVars(tasks, vtype=GRB.CONTINUOUS, name="t")
        T = model.addVar(vtype=GRB.CONTINUOUS, name="T")

        # 保存变量引用
        order_vars: Dict[Tuple[str, str, str], gp.Var] = {}
        region_order_vars: Dict[Tuple[str, str], gp.Var] = {}

        # 约束 1. 每个任务必须且只能被分配给一只手臂
        for i in tasks:
            model.addConstr(gp.quicksum(x[i, a] for a in accessibility[i]) == 1, name=f"assign_{i}")

        # 约束 2. 遵守可达性
        for i in tasks:
            for a in arms:
                if a not in accessibility[i]:
                    model.addConstr(x[i, a] == 0, name=f"reach_{i}_{a}")

        # 约束 3. 同一手臂的顺序约束
        for idx_i in range(len(tasks)):
            for idx_j in range(idx_i + 1, len(tasks)):
                i = tasks[idx_i]
                j = tasks[idx_j]

                for a in arms:
                    if a in accessibility[i] and a in accessibility[j]:
                        o_ij = model.addVar(vtype=GRB.BINARY, name=f"order_{i}_{j}_{a}")
                        o_ji = model.addVar(vtype=GRB.BINARY, name=f"order_{j}_{i}_{a}")
                        # 保存引用，供热启动直接使用
                        order_vars[(i, j, a)] = o_ij
                        order_vars[(j, i, a)] = o_ji

                        model.addConstr(o_ij + o_ji == 1, name=f"order_sum_{i}_{j}_{a}")

                        M = 1000
                        model.addConstr(
                            t[i] + processing_time[i, a]
                            <= t[j] + M * (1 - o_ij) + M * (1 - x[i, a]) + M * (1 - x[j, a]),
                            name=f"seq_{i}_{j}_{a}"
                        )
                        model.addConstr(
                            t[j] + processing_time[j, a]
                            <= t[i] + M * (1 - o_ji) + M * (1 - x[i, a]) + M * (1 - x[j, a]),
                            name=f"seq_{j}_{i}_{a}"
                        )

        # 约束 4. 干涉区内的任务必须串行
        # 按区域动态分组，不再硬编码区域名（如原来的 b2_tasks / b5_tasks）
        interference_by_region: Dict[str, List[str]] = defaultdict(list)
        for i in interference_tasks:
            interference_by_region[self.milp_params["regions"][i]].append(i)

        for region_name, region_task_list in interference_by_region.items():
            for idx_i in range(len(region_task_list)):
                for idx_j in range(idx_i + 1, len(region_task_list)):
                    i = region_task_list[idx_i]
                    j = region_task_list[idx_j]

                    y_ij = model.addVar(vtype=GRB.BINARY, name=f"region_order_{i}_{j}")
                    y_ji = model.addVar(vtype=GRB.BINARY, name=f"region_order_{j}_{i}")
                    # 保存引用，供热启动直接使用
                    region_order_vars[(i, j)] = y_ij
                    region_order_vars[(j, i)] = y_ji

                    model.addConstr(y_ij + y_ji == 1, name=f"region_order_sum_{i}_{j}")

                    M = 1000
                    for a1 in accessibility[i]:
                        for a2 in accessibility[j]:
                            model.addConstr(
                                t[i] + processing_time[i, a1]
                                <= t[j] + M * (1 - y_ij) + M * (1 - x[i, a1]) + M * (1 - x[j, a2]),
                                name=f"region_seq_{i}_{j}_{a1}_{a2}"
                            )

                    for a1 in accessibility[j]:
                        for a2 in accessibility[i]:
                            model.addConstr(
                                t[j] + processing_time[j, a1]
                                <= t[i] + M * (1 - y_ji) + M * (1 - x[j, a1]) + M * (1 - x[i, a2]),
                                name=f"region_seq_{j}_{i}_{a1}_{a2}"
                            )

        # 约束 5. 总完工时间 T 必须大于等于所有任务的结束时间
        for i in tasks:
            for a in accessibility[i]:
                model.addConstr(
                    t[i] + processing_time[i, a] <= T + 1000 * (1 - x[i, a]),
                    name=f"makespan_{i}_{a}"
                )

        # 约束 6. 最小化 T（总耗时）
        model.setObjective(T, GRB.MINIMIZE)

        # 把变量引用挂到 model 上
        model._varmap = {
            "x": x,
            "t": t,
            "T": T,
            "order": order_vars,
            "region_order": region_order_vars,
            "tasks": tasks,
            "arms": arms,
        }

        # 如果提供，设置热启动
        if warm_start_actions is not None:
            self._set_warm_start(model, warm_start_actions)

        return model

    # ----------------------------------------------------------------------- #
    # 核心方法：解提取                                                         #
    # ----------------------------------------------------------------------- #

    def _extract_solution_vars(self, model: gp.Model) -> Tuple[Dict, Dict]:
        """从求解的模型中提取解变量。"""
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
        """从解变量中提取动作序列。"""
        actions = []
        for task_id in t_vals:
            row = self.task_df.iloc[int(task_id[1:])]
            arm = 'L' if x_vals.get((task_id, 'L'), 0) > 0.5 else 'R'
            start_time = t_vals[task_id]
            end_time = start_time + row[f'time_to_{arm}']
            actions.append({
                'task': task_id,
                'arm': arm,
                'x': row['x'],
                'y': row['y'],
                'start': start_time,
                'end': end_time
            })
        return sorted(actions, key=lambda a: a['start'])

    # ----------------------------------------------------------------------- #
    # 核心方法：热启动                                                         #
    # ----------------------------------------------------------------------- #

    def _set_warm_start(self, model: gp.Model, actions: List[Dict]):
        """
        根据启发式动作设置热启动值。

        干涉区按区域动态分组，不再硬编码区域名（如原来的 b2_tasks/b5_tasks）。
        """
        try:
            if not hasattr(model, "_varmap") or model._varmap is None:
                raise RuntimeError("Warm start (B) requires model._varmap. Build the model with variable mapping first.")

            varmap = model._varmap
            x = varmap["x"]
            t = varmap["t"]
            T = varmap["T"]
            order_vars: Dict[Tuple[str, str, str], gp.Var] = varmap.get("order", {})
            region_order_vars: Dict[Tuple[str, str], gp.Var] = varmap.get("region_order", {})

            tasks = self.milp_params["tasks"]
            accessibility = self.milp_params["accessibility"]

            # ---- 1) 从 actions 解析出 x_warm / t_warm / T_warm ----
            x_warm: Dict[Tuple[str, str], int] = {}
            t_warm: Dict[str, float] = {}

            # 初始化：所有可达 (task, arm) 置 0
            for task_id in tasks:
                for arm in ["L", "R"]:
                    if arm in accessibility[task_id]:
                        x_warm[(task_id, arm)] = 0

            # 写入启发式动作
            for a in actions:
                task_id = a["task"]
                arm = a["arm"]
                start_time = float(a["start"])
                x_warm[(task_id, arm)] = 1
                t_warm[task_id] = start_time

            T_warm = float(max(a["end"] for a in actions)) if actions else 0.0

            # ---- 2) 设置 Start：x / t / T ----
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

            # ---- 3) order 变量：先给一个稳定默认顺序，再用启发式顺序覆盖（提升接受概率）----
            task_index = {tid: idx for idx, tid in enumerate(tasks)}

            # 默认：按 tasks 列表顺序（idx 小的在前）
            for (i, j, arm), v in order_vars.items():
                if task_index.get(i, 0) < task_index.get(j, 0):
                    v.Start = 1
                else:
                    v.Start = 0

            # 覆盖：同一手臂上，按启发式开始时间排序
            for arm in ["L", "R"]:
                arm_actions = sorted([a for a in actions if a["arm"] == arm], key=lambda z: z["start"])
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

            # ---- 4) region_order 变量：先默认，再用启发式干涉区开始时间覆盖 ----
            for (i, j), v in region_order_vars.items():
                if task_index.get(i, 0) < task_index.get(j, 0):
                    v.Start = 1
                else:
                    v.Start = 0

            # 按区域动态分组，不再硬编码区域名
            interference_by_region: Dict[str, List[str]] = defaultdict(list)
            for tid in self.milp_params["interference_set"]:
                interference_by_region[self.milp_params["regions"][tid]].append(tid)

            for region_name, region_task_list in interference_by_region.items():
                task_times = [(tid, t_warm[tid]) for tid in region_task_list if tid in t_warm]
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

            print(f"Warm start values set successfully (B scheme). Heuristic makespan: {T_warm:.2f}s")

        except Exception as e:
            print(f"Warning: Could not set warm start values (B scheme): {e}")

    # ----------------------------------------------------------------------- #
    # 核心方法：结果保存                                                       #
    # ----------------------------------------------------------------------- #

    def save_results(self, result_dir: str = "result") -> None:
        """
        保存优化结果，包括动画、图表和摘要。

        参数:
            result_dir: 保存结果的目录
        """
        if self.heuristic_actions is None:
            raise ValueError("No results to save. Run solve_optimization() first.")

        os.makedirs(result_dir, exist_ok=True)

        # 保存启发式结果
        heuristic_makespan = max(action["end"] for action in self.heuristic_actions)
        self.animate_solution(self.heuristic_actions,
                              save_path=f"{result_dir}/heuristic_animation.gif")
        self.plot_gantt_chart(self.heuristic_actions,
                              save_path=f"{result_dir}/heuristic_gantt.png",
                              title_suffix=f" - Heuristic (Makespan: {heuristic_makespan:.2f}s)")

        # 如果可用，保存 MILP 结果
        if self.milp_actions is not None:
            milp_makespan = max(action["end"] for action in self.milp_actions)
            self.animate_solution(self.milp_actions,
                                  save_path=f"{result_dir}/milp_animation.gif")
            self.plot_gantt_chart(self.milp_actions,
                                  save_path=f"{result_dir}/milp_gantt.png",
                                  title_suffix=f" - MILP Optimized (Makespan: {milp_makespan:.2f}s)")

            # 保存摘要
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

    # ----------------------------------------------------------------------- #
    # 核心方法：绘图骨架                                                       #
    # ----------------------------------------------------------------------- #

    def animate_solution(self, actions: List[Dict], save_path: Optional[str] = None,
                         title: str = "Dual-Arm Harvesting Animation") -> animation.FuncAnimation:
        """
        创建采摘解决方案的动画。

        通过 ``_draw_environment_background`` 和 ``_get_task_color`` 钩子
        适配不同模式的环境背景与配色规则。

        参数:
            actions: 动作字典列表
            save_path: 将动画保存为 GIF 的可选路径
            title: 动画标题
        返回:
            动画对象
        """
        if self.task_df is None:
            raise ValueError("No task data loaded.")

        fig, ax = plt.subplots(figsize=(8, 8))

        # 钩子：绘制环境背景（区域边界、操作边界等）
        self._draw_environment_background(ax, with_region_labels=True)

        # 绘制手臂基座
        arm_L, = ax.plot(*self.L_base, 'ks', markersize=10, label='Left Arm Base (L)')
        arm_R, = ax.plot(*self.R_base, 'ko', markersize=10, label='Right Arm Base (R)')

        # 设置绘图
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_title(title)

        # 初始化任务点（颜色由钩子决定）
        task_points = {}
        task_states = {}

        for act in actions:
            task_id = act['task']
            color = self._get_task_color(task_id)
            point, = ax.plot(act['x'], act['y'], 'o', color=color, markersize=8, alpha=0.8)
            task_points[task_id] = point
            task_states[task_id] = False

        # 手臂连线和时间显示
        line_L, = ax.plot([], [], color='blue', lw=3, alpha=0.8, label='Left Arm')
        line_R, = ax.plot([], [], color='red', lw=3, alpha=0.8, label='Right Arm')
        time_text = ax.text(-0.95, 0.95, '', fontsize=14,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax.legend(loc='upper right')

        def init():
            line_L.set_data([], [])
            line_R.set_data([], [])
            time_text.set_text('')
            return [arm_L, arm_R, line_L, line_R, time_text] + list(task_points.values())

        def update(frame):
            t = frame * 0.1  # 100ms 间隔
            time_text.set_text(f"Time: {t:.1f}s")

            line_L.set_data([], [])
            line_R.set_data([], [])

            for act in actions:
                task_id = act['task']

                if act['start'] <= t <= act['end']:
                    # 手臂正在执行任务
                    base = self.L_base if act['arm'] == 'L' else self.R_base
                    if act['arm'] == 'L':
                        line_L.set_data([base[0], act['x']], [base[1], act['y']])
                    else:
                        line_R.set_data([base[0], act['x']], [base[1], act['y']])

                elif t > act['end'] and not task_states[task_id]:
                    # 任务完成
                    task_states[task_id] = True
                    task_points[task_id].remove()

                    color = self._get_task_color(task_id)
                    new_point, = ax.plot(act['x'], act['y'], 'x', color=color,
                                        markersize=12, markeredgewidth=3)
                    task_points[task_id] = new_point

            return [arm_L, arm_R, line_L, line_R, time_text] + list(task_points.values())

        # 创建动画
        max_time = max(act['end'] for act in actions) if actions else 10.0
        frames = int(max_time * 10) + 10  # 100ms 间隔

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

    def plot_gantt_chart(self, actions: List[Dict], save_path: Optional[str] = None,
                         title_suffix: str = "") -> None:
        """
        创建甘特图可视化。

        任务颜色通过 ``_get_task_color`` 钩子决定；
        图例条目从 task_df 动态构建，不再使用硬编码的 region_colors 字典。

        参数:
            actions: 动作字典列表
            save_path: 将图表保存为 PNG 的可选路径
            title_suffix: 标题的附加文本
        """
        if self.task_df is None:
            raise ValueError("No task data loaded.")

        fig, ax = plt.subplots(figsize=(12, 6))

        left_actions = [a for a in actions if a['arm'] == 'L']
        right_actions = [a for a in actions if a['arm'] == 'R']

        # 绘制左臂任务
        for action in left_actions:
            task_id = action['task']
            task_idx = int(task_id[1:])
            region = self.task_df.iloc[task_idx]['region']
            color = self._get_task_color(task_id)

            ax.barh(0, action['end'] - action['start'], left=action['start'],
                    height=0.4, color=color, alpha=0.7, edgecolor='black')

            mid_time = (action['start'] + action['end']) / 2
            ax.text(mid_time, 0, f"{task_id}\n{region}", ha='center', va='center',
                    fontsize=8, fontweight='bold')

        # 绘制右臂任务
        for action in right_actions:
            task_id = action['task']
            task_idx = int(task_id[1:])
            region = self.task_df.iloc[task_idx]['region']
            color = self._get_task_color(task_id)

            ax.barh(1, action['end'] - action['start'], left=action['start'],
                    height=0.4, color=color, alpha=0.7, edgecolor='black')

            mid_time = (action['start'] + action['end']) / 2
            ax.text(mid_time, 1, f"{task_id}\n{region}", ha='center', va='center',
                    fontsize=8, fontweight='bold')

        # 自定义绘图
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Left Arm (L)', 'Right Arm (R)'])
        ax.set_xlabel('Time (seconds)')
        ax.set_title(f'Dual-Arm Task Scheduling Gantt Chart{title_suffix}')
        ax.grid(True, axis='x', alpha=0.3)

        # 从 task_df 动态构建图例（不再硬编码 region_colors 字典）
        region_to_color: Dict[str, str] = {}
        for idx, row in self.task_df.iterrows():
            region = row['region']
            if region not in region_to_color:
                region_to_color[region] = self._get_task_color(f"t{idx}")

        legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.7,
                                         edgecolor='black', label=region)
                           for region, color in sorted(region_to_color.items())]
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

    # ----------------------------------------------------------------------- #
    # 核心方法：综合结果保存                                                   #
    # ----------------------------------------------------------------------- #

    def save_comprehensive_results(self, result_dir: str = "comprehensive_results",
                                   experiment_name: str = None,
                                   include_data: bool = True,
                                   include_config: bool = True) -> None:
        """
        使用详细分析和有组织的文件夹结构保存综合优化结果。

        参数:
            result_dir: 保存结果的基础目录
            experiment_name: 可选实验名称（如果为 None 则自动生成）
            include_data: 是否保存原始数据文件
            include_config: 是否保存配置和参数
        """
        if self.heuristic_actions is None:
            raise ValueError("No results to save. Run solve_optimization() first.")

        # 如果未提供，则生成实验名称
        if experiment_name is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            experiment_name = f"dual_arm_experiment_{timestamp}"

        # 创建有组织的文件夹结构
        base_dir = os.path.join(result_dir, experiment_name)
        folders = {
            'animations': os.path.join(base_dir, 'animations'),
            'plots': os.path.join(base_dir, 'plots'),
            'data': os.path.join(base_dir, 'data'),
            'analysis': os.path.join(base_dir, 'analysis'),
            'config': os.path.join(base_dir, 'config')
        }

        for folder in folders.values():
            os.makedirs(folder, exist_ok=True)

        print(f"Saving comprehensive results to: {base_dir}")

        # 计算指标
        heuristic_makespan = max(action["end"] for action in self.heuristic_actions)

        milp_makespan = None
        if self.milp_actions is not None:
            milp_makespan = max(action["end"] for action in self.milp_actions)

        # 1. 保存动画
        print("  \u2192 Generating animations...")
        self.animate_solution(
            self.heuristic_actions,
            save_path=os.path.join(folders['animations'], 'heuristic_animation.gif'),
            title=f"Heuristic Solution - Makespan: {heuristic_makespan:.2f}s"
        )

        if self.milp_actions is not None:
            self.animate_solution(
                self.milp_actions,
                save_path=os.path.join(folders['animations'], 'milp_animation.gif'),
                title=f"MILP Solution - Makespan: {milp_makespan:.2f}s"
            )

        # 2. 保存甘特图
        print("  \u2192 Creating Gantt charts...")
        self.plot_gantt_chart(
            self.heuristic_actions,
            save_path=os.path.join(folders['plots'], 'heuristic_gantt.png'),
            title_suffix=f" - Heuristic (Makespan: {heuristic_makespan:.2f}s)"
        )

        if self.milp_actions is not None:
            self.plot_gantt_chart(
                self.milp_actions,
                save_path=os.path.join(folders['plots'], 'milp_gantt.png'),
                title_suffix=f" - MILP Optimized (Makespan: {milp_makespan:.2f}s)"
            )

        # 3. 保存任务分布图
        print("  \u2192 Creating task distribution plot...")
        self._plot_task_distribution(os.path.join(folders['plots'], 'task_distribution.png'))

        # 4. 保存区域利用率分析
        print("  \u2192 Creating region utilization analysis...")
        self._plot_region_utilization(folders['plots'])

        # 5. 保存原始数据（如果请求）
        if include_data:
            print("  \u2192 Saving raw data...")
            self._save_raw_data(folders['data'])

        # 6. 保存配置（如果请求）
        if include_config:
            print("  \u2192 Saving configuration...")
            self._save_configuration(folders['config'])

        # 7. 保存综合摘要
        print("  \u2192 Creating comprehensive summary...")
        self._save_comprehensive_summary(base_dir, heuristic_makespan, milp_makespan)

        # 8. 保存性能比较
        if self.milp_actions is not None:
            print("  \u2192 Creating performance comparison...")
            self._create_performance_comparison(folders['analysis'])

        print(f"\u2713 Comprehensive results saved successfully!")
        print(f"  Main directory: {base_dir}")
        print(f"  Animations: {folders['animations']}")
        print(f"  Plots: {folders['plots']}")
        print(f"  Analysis: {folders['analysis']}")

        # 关闭所有 matplotlib 图形以防止内存问题
        plt.close('all')

    def _plot_task_distribution(self, save_path: str) -> None:
        """创建显示各区域任务分布的图。"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # 按区域统计任务数量
        region_counts = self.task_df['region'].value_counts().sort_index()

        # 通过 _get_task_color 钩子获取每个区域的代表色
        bar_colors = []
        for region_name in region_counts.index:
            region_tasks = self.task_df[self.task_df['region'] == region_name]
            first_idx = region_tasks.index[0]
            bar_colors.append(self._get_task_color(f"t{first_idx}"))

        ax1.bar(region_counts.index, region_counts.values, color=bar_colors, alpha=0.7, edgecolor='black')
        ax1.set_title('Task Count by Region')
        ax1.set_xlabel('Region')
        ax1.set_ylabel('Number of Tasks')
        ax1.grid(True, alpha=0.3)

        # 任务位置散点图
        for region_name in sorted(self.task_df['region'].unique()):
            region_tasks = self.task_df[self.task_df['region'] == region_name]
            if not region_tasks.empty:
                first_idx = region_tasks.index[0]
                color = self._get_task_color(f"t{first_idx}")
                ax2.scatter(region_tasks['x'], region_tasks['y'],
                            color=color, label=region_name, s=50, alpha=0.7)

        # 钩子：绘制区域边界（不添加区域标签以避免与散点图例重复）
        self._draw_environment_background(ax2, with_region_labels=False)

        # 绘制手臂基座
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
        """创建区域利用率分析图。"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))

        solutions = [("Heuristic", self.heuristic_actions), ("MILP", self.milp_actions)]

        for idx, (method, actions) in enumerate(solutions):
            if actions is None:
                continue

            ax = [ax1, ax2][idx]

            # 计算每个区域的总时间
            region_times = {"L": {}, "R": {}}

            for action in actions:
                arm = action["arm"]
                task_idx = int(action["task"][1:])
                region = self.task_df.iloc[task_idx]["region"]

                if region not in region_times[arm]:
                    region_times[arm][region] = 0

                region_times[arm][region] += action["end"] - action["start"]

            # 绘制条形图（区域名从 task_df 派生）
            regions = sorted(self.task_df['region'].unique().tolist())
            left_times = [region_times["L"].get(region, 0) for region in regions]
            right_times = [region_times["R"].get(region, 0) for region in regions]

            x = np.arange(len(regions))
            width = 0.35

            ax.bar(x - width / 2, left_times, width, label='Left Arm', color='blue', alpha=0.7)
            ax.bar(x + width / 2, right_times, width, label='Right Arm', color='red', alpha=0.7)

            ax.set_xlabel('Region')
            ax.set_ylabel('Total Time (s)')
            ax.set_title(f'Region Utilization - {method}')
            ax.set_xticks(x)
            ax.set_xticklabels(regions)
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 两个手臂的工作量比较
        if self.milp_actions is not None:
            methods = ["Heuristic", "MILP"]
            solutions = [self.heuristic_actions, self.milp_actions]

            left_workloads = []
            right_workloads = []

            for actions in solutions:
                left_time = sum(action["end"] - action["start"]
                                for action in actions if action["arm"] == "L")
                right_time = sum(action["end"] - action["start"]
                                 for action in actions if action["arm"] == "R")
                left_workloads.append(left_time)
                right_workloads.append(right_time)

            x = np.arange(len(methods))
            width = 0.35

            ax3.bar(x - width / 2, left_workloads, width, label='Left Arm', color='blue', alpha=0.7)
            ax3.bar(x + width / 2, right_workloads, width, label='Right Arm', color='red', alpha=0.7)
            ax3.set_xlabel('Solution Method')
            ax3.set_ylabel('Total Workload (s)')
            ax3.set_title('Arm Workload Comparison')
            ax3.set_xticks(x)
            ax3.set_xticklabels(methods)
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # 添加值标签
            for i, (left, right) in enumerate(zip(left_workloads, right_workloads)):
                ax3.text(i - width / 2, left + 0.1, f'{left:.1f}', ha='center', va='bottom')
                ax3.text(i + width / 2, right + 0.1, f'{right:.1f}', ha='center', va='bottom')

        # 时间线可视化
        if self.milp_actions is not None:
            self._plot_arm_timeline(ax4)

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'region_utilization.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

    def _plot_arm_timeline(self, ax) -> None:
        """绘制手臂时间线图。"""
        actions = self.milp_actions if self.milp_actions else self.heuristic_actions

        left_actions = [a for a in actions if a['arm'] == 'L']
        right_actions = [a for a in actions if a['arm'] == 'R']

        for action in left_actions:
            ax.barh(0, action['end'] - action['start'], left=action['start'],
                    height=0.3, color='blue', alpha=0.7)

        for action in right_actions:
            ax.barh(1, action['end'] - action['start'], left=action['start'],
                    height=0.3, color='red', alpha=0.7)

        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Left Arm', 'Right Arm'])
        ax.set_xlabel('Time (s)')
        ax.set_title('Arm Activity Timeline')
        ax.grid(True, alpha=0.3)

    def _save_raw_data(self, data_dir: str) -> None:
        """保存原始数据文件。"""
        self.task_df.to_csv(os.path.join(data_dir, 'task_data.csv'), index=False)

        if self.heuristic_actions:
            heuristic_df = pd.DataFrame(self.heuristic_actions)
            heuristic_df.to_csv(os.path.join(data_dir, 'heuristic_actions.csv'), index=False)

        if self.milp_actions:
            milp_df = pd.DataFrame(self.milp_actions)
            milp_df.to_csv(os.path.join(data_dir, 'milp_actions.csv'), index=False)

    def _save_configuration(self, config_dir: str) -> None:
        """
        保存配置和参数。

        干涉区信息从 task_df 的 is_interference 字段派生，
        不依赖 self.interference_regions 等硬编码属性。
        子类可以覆盖此方法以添加模式相关的配置（如 regions、region_colors）。
        """
        import json

        interference_regions = sorted(
            self.task_df[self.task_df['is_interference']]['region'].unique().tolist()
        ) if self.task_df is not None else []

        config = {
            "arm_bases": {
                "left": self.L_base.tolist(),
                "right": self.R_base.tolist()
            },
            "interference_regions": interference_regions,
            "total_tasks": len(self.task_df) if self.task_df is not None else 0,
        }

        with open(os.path.join(config_dir, 'experiment_config.json'), 'w') as f:
            json.dump(config, f, indent=2)

    def _save_comprehensive_summary(self, base_dir: str, heuristic_makespan: float,
                                    milp_makespan: float) -> None:
        """保存综合分析摘要。"""
        summary_path = os.path.join(base_dir, 'EXPERIMENT_SUMMARY.md')

        # 从 task_df 派生区域信息，不依赖 self.regions 或 self.interference_regions
        all_regions = sorted(self.task_df['region'].unique().tolist())
        interference_regions = sorted(
            self.task_df[self.task_df['is_interference']]['region'].unique().tolist()
        ) if self.task_df is not None else []

        with open(summary_path, 'w') as f:
            f.write("# Dual-Arm Harvesting Optimization Results\n\n")
            f.write(f"**Experiment Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Problem Configuration\n")
            f.write(f"- **Total Tasks**: {len(self.task_df)}\n")
            f.write(f"- **Regions**: {', '.join(all_regions)}\n")
            f.write(f"- **Interference Regions**: {', '.join(interference_regions)}\n")
            f.write(f"- **Left Arm Base**: ({self.L_base[0]:.2f}, {self.L_base[1]:.2f})\n")
            f.write(f"- **Right Arm Base**: ({self.R_base[0]:.2f}, {self.R_base[1]:.2f})\n\n")

            f.write("## Task Distribution by Region\n")
            region_counts = self.task_df['region'].value_counts().sort_index()
            for region, count in region_counts.items():
                f.write(f"- **{region}**: {count} tasks\n")
            f.write("\n")

            f.write("## Solution Performance\n\n")
            f.write("| Method | Makespan (s) |\n")
            f.write("|--------|--------------|\n")
            f.write(f"| Heuristic | {heuristic_makespan:.2f} |\n")

            if milp_makespan is not None:
                f.write(f"| MILP | {milp_makespan:.2f} |\n")
                f.write(f"| **Improvement** | **{self.improvement:.1f}%** |\n\n")
            else:
                f.write("| MILP | No solution found |\n\n")

            f.write("## Files Generated\n\n")
            f.write("### Animations\n")
            f.write("- `animations/heuristic_animation.gif` - Heuristic solution animation\n")
            if milp_makespan is not None:
                f.write("- `animations/milp_animation.gif` - MILP solution animation\n")

            f.write("\n### Plots\n")
            f.write("- `plots/heuristic_gantt.png` - Heuristic Gantt chart\n")
            if milp_makespan is not None:
                f.write("- `plots/milp_gantt.png` - MILP Gantt chart\n")
            f.write("- `plots/task_distribution.png` - Task distribution analysis\n")
            f.write("- `plots/region_utilization.png` - Region utilization analysis\n")

            f.write("\n### Data\n")
            f.write("- `data/task_data.csv` - Raw task data\n")
            f.write("- `data/heuristic_actions.csv` - Heuristic solution actions\n")
            if milp_makespan is not None:
                f.write("- `data/milp_actions.csv` - MILP solution actions\n")

            f.write("\n### Configuration\n")
            f.write("- `config/experiment_config.json` - Experiment configuration\n")

            if milp_makespan is not None:
                f.write("\n## Key Insights\n\n")
                if self.improvement > 5:
                    f.write(f"- **Significant Improvement**: MILP achieved {self.improvement:.1f}% reduction in makespan\n")
                elif self.improvement > 0:
                    f.write(f"- **Minor Improvement**: MILP achieved {self.improvement:.1f}% reduction in makespan\n")
                else:
                    f.write("- **Heuristic Optimal**: MILP could not improve upon the heuristic solution\n")

        print(f"  \u2713 Summary saved to: EXPERIMENT_SUMMARY.md")

    def _create_performance_comparison(self, analysis_dir: str) -> None:
        """创建性能比较图表。"""
        if self.milp_actions is None:
            return

        fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(16, 6))

        # 1. Makespan comparison
        methods = ['Heuristic', 'MILP']
        makespans = [
            max(action["end"] for action in self.heuristic_actions),
            max(action["end"] for action in self.milp_actions)
        ]

        bars1 = ax1.bar(methods, makespans, color=['lightblue', 'lightgreen'],
                        edgecolor='black', alpha=0.7)
        ax1.set_ylabel('Makespan (seconds)')
        ax1.set_title('Makespan Comparison')
        ax1.grid(True, alpha=0.3)

        for bar, value in zip(bars1, makespans):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                     f'{value:.2f}s', ha='center', va='bottom', fontweight='bold')

        improvement = ((makespans[0] - makespans[1]) / makespans[0]) * 100
        ax1.text(0.5, max(makespans) * 0.9, f'Improvement: {improvement:.1f}%',
                 ha='center', transform=ax1.transData, fontsize=12, fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

        # 2. Task completion timeline
        for i, (method, actions, color) in enumerate([
            ('Heuristic', self.heuristic_actions, 'blue'),
            ('MILP', self.milp_actions, 'green')
        ]):
            completion_times = sorted([action["end"] for action in actions])
            task_numbers = list(range(1, len(completion_times) + 1))
            ax3.plot(completion_times, task_numbers, marker='o', label=method,
                     color=color, alpha=0.7, linewidth=2)

        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Tasks Completed')
        ax3.set_title('Task Completion Timeline')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(analysis_dir, 'performance_comparison.png'),
                    dpi=300, bbox_inches='tight')
        plt.close(fig)


# ===========================================================================
# Baseline 规划器（Baseline Planner）
# 包含：B1-B6 区域配置、欧氏距离处理时间、基于区域的任务生成/加载、
#       空间顺序启发式、Baseline 环境背景绘制与配色规则
# ===========================================================================

class BaselinePlanner(DualArmPlannerCore):
    """
    Baseline 规划器。

    继承 DualArmPlannerCore，实现 Baseline 模式的专属逻辑：
      - B1-B6 六区域配置（双臂可达的区域自动识别为干涉区）
      - 欧氏距离近似单程移动时间
      - 基于区域的随机任务生成（create_task_dataset）与自定义加载（load_task_locations）
      - 空间顺序启发式（左臂 B2→B1→B4，右臂 B5→B6→B3）
      - 区域矩形背景绘制（_draw_environment_background）
      - 按区域索引的配色方案（_get_task_color）

    task_df 中每行携带 ``is_interference`` 字段（bool），
    由区域配置自动生成，为核心类所有约束和逻辑的统一依据。
    """

    _DEFAULT_REGIONS: List[Dict] = [
        {"center": (-0.4,  0.5), "width": 0.4, "height": 0.4, "name": "B1", "arm_access": ["L"]},
        {"center": ( 0.0,  0.5), "width": 0.4, "height": 0.4, "name": "B2", "arm_access": ["L", "R"]},
        {"center": ( 0.4,  0.5), "width": 0.4, "height": 0.4, "name": "B3", "arm_access": ["R"]},
        {"center": (-0.4, -0.5), "width": 0.4, "height": 0.4, "name": "B4", "arm_access": ["L"]},
        {"center": ( 0.0, -0.5), "width": 0.4, "height": 0.4, "name": "B5", "arm_access": ["L", "R"]},
        {"center": ( 0.4, -0.5), "width": 0.4, "height": 0.4, "name": "B6", "arm_access": ["R"]},
    ]
    _DEFAULT_REGION_COLORS: List[str] = ["red", "orange", "green", "blue", "purple", "brown"]

    def __init__(self,
                 L_base: np.ndarray = np.array([-0.3, 0]),
                 R_base: np.ndarray = np.array([0.3, 0]),
                 regions_config: Optional[List[Dict]] = None,
                 base_operation_time: float = 0.0):
        """
        参数:
            L_base: 左臂基座位置 (x, y)
            R_base: 右臂基座位置 (x, y)
            regions_config: 采摘区域配置。若为 None 则使用默认 B1-B6 六区域。
            base_operation_time: 单个草莓的固定处理时间（包括采摘、放置等）
        """
        super().__init__(L_base=L_base, R_base=R_base, base_operation_time=base_operation_time)

        self.regions = self._DEFAULT_REGIONS if regions_config is None else regions_config
        self.region_colors = list(self._DEFAULT_REGION_COLORS)

        # 干涉区由区域配置自动识别（双臂可达 → 干涉区），不再硬编码区域名
        self._interference_region_names: set = {
            r["name"] for r in self.regions if len(r["arm_access"]) > 1
        }

    # ----------------------------------------------------------------------- #
    # Baseline 数据生成                                                        #
    # ----------------------------------------------------------------------- #

    def create_task_dataset(self, points_per_region: Dict[str, int]) -> pd.DataFrame:
        """
        创建随机草莓任务数据集。

        task_df 中每行携带 ``is_interference`` 字段，
        由该区域是否属于干涉区自动填充。

        输入：每个区域的任务点数量
        输出：每个草莓的具体位置、对左右臂的可达性、处理时间及干涉区标志
        """
        task_data = []

        for region in self.regions:
            cx, cy = region["center"]
            w, h = region["width"], region["height"]
            arm_access = region["arm_access"]
            name = region["name"]
            is_interference = name in self._interference_region_names

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
                    "is_interference": is_interference,
                    "time_to_L": time_L,
                    "time_to_R": time_R,
                })

        self.task_df = pd.DataFrame(task_data)
        self._extract_milp_parameters()
        return self.task_df

    def generate_random_points(self, center: Tuple[float, float], width: float,
                               height: float, n: int = 5) -> np.ndarray:
        """在指定矩形范围内生成随机点。"""
        cx, cy = center
        xs = np.random.uniform(cx - width / 2, cx + width / 2, n)
        ys = np.random.uniform(cy - height / 2, cy + height / 2, n)
        return np.vstack((xs, ys)).T

    def load_task_locations(self, task_locations: List[Dict]) -> pd.DataFrame:
        """
        从字典列表中加载任务位置。

        task_df 中每行携带 ``is_interference`` 字段，
        由该区域是否属于干涉区自动填充。
        """
        task_data = []

        for task in task_locations:
            x = float(task["x"])
            y = float(task["y"])
            region_name = task["region"]

            region_config = next((r for r in self.regions if r["name"] == region_name), None)
            if region_config is None:
                raise ValueError(f"Region {region_name} not found in configuration.")
            arm_access = region_config["arm_access"]
            is_interference = region_name in self._interference_region_names

            pt = np.array([x, y])
            time_L = self._compute_processing_time(pt, "L") if "L" in arm_access else None
            time_R = self._compute_processing_time(pt, "R") if "R" in arm_access else None

            task_data.append({
                "region": region_name,
                "x": x,
                "y": y,
                "accessible_by": arm_access,
                "is_interference": is_interference,
                "time_to_L": time_L,
                "time_to_R": time_R,
            })

        self.task_df = pd.DataFrame(task_data)
        self._extract_milp_parameters()
        return self.task_df

    def _compute_processing_time(self, point: np.ndarray, arm: str) -> float:
        """
        计算采摘时间：2 × 单程移动时间（欧式距离近似）+ 固定处理时间。

        参数:
            point: 草莓位置 [x, y]
            arm: 'L' 或 'R'
        """
        base = self.L_base if arm == "L" else self.R_base
        one_way_time = float(np.linalg.norm(point - base))
        return 2 * one_way_time + self.base_operation_time

    # ----------------------------------------------------------------------- #
    # Baseline 启发式                                                          #
    # ----------------------------------------------------------------------- #

    def spatial_order_heuristic(self) -> List[Dict]:
        """
        基于区域的空间顺序启发式算法。

        左臂: B2 → B1 → B4（先干涉区，再自有区）
        右臂: B5 → B6 → B3（先干涉区，再自有区）

        干涉区判断使用 task_df 的 ``is_interference`` 字段，不再硬编码区域名。
        """
        if self.task_df is None:
            raise ValueError("No task data loaded.")

        actions = []
        arm_times = {'L': 0.0, 'R': 0.0}

        # 从 task_df 动态构建干涉区 busy_until 表
        interference_regions = set(
            self.task_df[self.task_df['is_interference']]['region'].unique()
        )
        region_busy_until = {r: 0.0 for r in interference_regions}

        # 每个手臂优先处理其干涉区域
        left_arm_order = ['B2', 'B1', 'B4']
        right_arm_order = ['B5', 'B6', 'B3']

        def schedule_region_tasks(arm, region, start_time):
            """处理单个区域的核心调度器。"""
            region_tasks = self.task_df[
                (self.task_df['region'] == region) &
                (self.task_df['accessible_by'].apply(lambda x: arm in x))
            ].copy()

            if region_tasks.empty:
                return [], start_time

            # 按处理时间升序排列（贪心策略：先采最快的）
            region_tasks["proc_time"] = region_tasks[f"time_to_{arm}"]
            region_tasks = region_tasks.sort_values("proc_time")
            current_time = start_time

            region_actions = []
            for idx, row in region_tasks.iterrows():
                if row['is_interference']:
                    # 干涉区：必须等上一个占用的任务释放该区域
                    actual_start = max(current_time, region_busy_until[region])
                    region_busy_until[region] = actual_start + row[f"time_to_{arm}"]
                else:
                    # 非干涉区：到场即可开工
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

        # 左臂调度流程
        left_time = arm_times['L']
        print("Left arm sequence: B2 -> B1 -> B4")
        for region in left_arm_order:
            region_actions, left_time = schedule_region_tasks('L', region, left_time)
            actions.extend(region_actions)
            arm_times['L'] = left_time

        # 右臂调度流程
        right_time = arm_times['R']
        print("Right arm sequence: B5 -> B6 -> B3")
        for region in right_arm_order:
            region_actions, right_time = schedule_region_tasks('R', region, right_time)
            actions.extend(region_actions)
            arm_times['R'] = right_time

        return sorted(actions, key=lambda x: x["start"])

    # ----------------------------------------------------------------------- #
    # Baseline 绘图钩子                                                        #
    # ----------------------------------------------------------------------- #

    def _draw_environment_background(self, ax, with_region_labels: bool = True) -> None:
        """
        绘制 B1-B6 区域边界及操作边界。

        参数:
            ax: matplotlib Axes 对象
            with_region_labels: 是否为区域添加图例标签。
                在动画中为 True（显示区域图例），
                在任务分布散点图中为 False（避免与散点标签重复）。
        """
        for i, region in enumerate(self.regions):
            cx, cy = region["center"]
            w, h = region["width"], region["height"]
            name = region["name"]
            color = self.region_colors[i]
            label = name if with_region_labels else '_nolegend_'
            ax.add_patch(plt.Rectangle((cx - w / 2, cy - h / 2), w, h,
                                       fill=False, edgecolor=color, linewidth=2, label=label))

        op_label = 'Operation Boundary' if with_region_labels else '_nolegend_'
        ax.add_patch(plt.Rectangle((-0.5, -0.25), 1.0, 0.5, fill=False,
                                   edgecolor='black', linestyle='--', linewidth=1.5,
                                   label=op_label))

    def _get_task_color(self, task_id: str) -> str:
        """根据任务所在区域返回对应的颜色。"""
        region_name = self.task_df.iloc[int(task_id[1:])]['region']
        region_idx = next(i for i, r in enumerate(self.regions) if r['name'] == region_name)
        return self.region_colors[region_idx]

    # ----------------------------------------------------------------------- #
    # Baseline 配置保存（覆盖核心类，添加区域相关字段）                       #
    # ----------------------------------------------------------------------- #

    def _save_configuration(self, config_dir: str) -> None:
        """保存实验配置，包含 Baseline 特有的区域及配色信息。"""
        import json

        interference_regions = sorted(list(self._interference_region_names))

        config = {
            "arm_bases": {
                "left": self.L_base.tolist(),
                "right": self.R_base.tolist()
            },
            "regions": self.regions,
            "interference_regions": interference_regions,
            "total_tasks": len(self.task_df) if self.task_df is not None else 0,
            "region_colors": self.region_colors
        }

        with open(os.path.join(config_dir, 'experiment_config.json'), 'w') as f:
            json.dump(config, f, indent=2)


# ===========================================================================
# RealCostPlanner：真实代价规划器
# 包含：从 URDF 读取基座位置、加载 dual_arm_cost.pkl、
#       从网格点采样任务、real-cost 贪心启发式、real-cost 绘图钩子
# ===========================================================================

class RealCostPlanner(DualArmPlannerCore):
    """
    真实代价规划器。

    继承 DualArmPlannerCore，接入 ``dual_arm_cost.pkl`` 的真实可达性与代价数据：

      - 从 URDF 文件解析左右臂基座位置（vehicle frame）
      - 加载 ``dual_arm_cost.pkl`` 中的 ``left_cost_table`` / ``right_cost_table``
      - 直接从 cost table 网格点采样任务（不做插值）
      - 生成 task_df，字段包含 region、x、y、z、table_key、
        accessible_by、is_interference、time_to_L、time_to_R
      - 提供简单贪心启发式（greedy heuristic）
      - 提供采样窗口背景绘图与按可达性着色

    干涉区定义：双臂都可达 → ``is_interference = True``。
    ``region`` 字段等于采样窗口名称，MILP 核心类通过 ``region`` 字段
    对干涉任务分组并施加串行约束（与 baseline 逻辑一致，无需修改 core）。

    采样窗口仅用于分组 / 采样 / 可视化，不人为定义干涉区，
    干涉标签完全由 cost table 真值决定。
    """

    # 默认采样窗口（与 dual_arm_cost.pkl 扫描范围匹配）
    # x 以 -0.15 为分界线，y 正负分左右侧果实带
    _DEFAULT_WINDOWS: List[Dict] = [
        {"name": "W_left_back",   "x_min": -0.65, "x_max": -0.15, "y_min":  0.30, "y_max":  0.60},
        {"name": "W_left_front",  "x_min": -0.15, "x_max":  0.30, "y_min":  0.30, "y_max":  0.60},
        {"name": "W_right_back",  "x_min": -0.65, "x_max": -0.15, "y_min": -0.60, "y_max": -0.30},
        {"name": "W_right_front", "x_min": -0.15, "x_max":  0.30, "y_min": -0.60, "y_max": -0.30},
    ]

    # 窗口背景填充色（顺序与 _DEFAULT_WINDOWS 对应）
    _WINDOW_FACECOLORS: List[str] = ["lightblue", "lightyellow", "lightsalmon", "lightgreen"]

    # 任务点配色（按可达性区分）
    _ACCESSIBILITY_COLORS: Dict[str, str] = {
        "left_only":  "steelblue",
        "right_only": "tomato",
        "both":       "darkorange",
    }

    def __init__(self,
                 urdf_path: str,
                 cost_pkl_path: str,
                 windows_config: Optional[List[Dict]] = None,
                 base_operation_time: float = 0.0):
        """
        参数:
            urdf_path: dual_arm_ik.urdf 文件路径（解析基座位置）
            cost_pkl_path: dual_arm_cost.pkl 文件路径
            windows_config: 采样窗口配置列表；None 则使用默认 4 窗口
            base_operation_time: 单草莓固定处理时间（采摘 + 放置，单位 s）
        """
        # 1. 从 URDF 解析左右臂基座 (x, y)
        L_xyz, R_xyz = self._load_base_xyz_from_urdf(urdf_path)
        L_base_2d = np.array([L_xyz[0], L_xyz[1]])
        R_base_2d = np.array([R_xyz[0], R_xyz[1]])

        super().__init__(L_base=L_base_2d, R_base=R_base_2d,
                         base_operation_time=base_operation_time)

        # 2. 加载 cost table
        self.cost_pkl_path = cost_pkl_path
        self._load_cost_tables(cost_pkl_path)

        # 3. 采样窗口
        self.windows = self._DEFAULT_WINDOWS if windows_config is None else windows_config

    # ----------------------------------------------------------------------- #
    # URDF 解析                                                                #
    # ----------------------------------------------------------------------- #

    @staticmethod
    def _load_base_xyz_from_urdf(urdf_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        从 URDF 解析左右臂基座在 vehicle frame 下的 xyz 位置。

        读取 ``vehicle_to_left_arm`` 和 ``vehicle_to_right_arm`` 固定关节的
        ``origin xyz`` 属性。
        """
        import xml.etree.ElementTree as ET

        tree = ET.parse(urdf_path)
        root = tree.getroot()

        def _get_origin_xyz(joint_name: str) -> np.ndarray:
            joint = root.find(f"./joint[@name='{joint_name}']")
            if joint is None:
                raise ValueError(f"Joint '{joint_name}' not found in {urdf_path}")
            origin = joint.find("origin")
            if origin is None:
                return np.zeros(3)
            xyz = [float(v) for v in origin.attrib.get("xyz", "0 0 0").split()]
            return np.array(xyz)

        L_xyz = _get_origin_xyz("vehicle_to_left_arm")
        R_xyz = _get_origin_xyz("vehicle_to_right_arm")
        return L_xyz, R_xyz

    # ----------------------------------------------------------------------- #
    # Cost table 加载                                                          #
    # ----------------------------------------------------------------------- #

    def _load_cost_tables(self, cost_pkl_path: str) -> None:
        """加载 dual_arm_cost.pkl，提取 left_cost_table 和 right_cost_table。"""
        import pickle

        with open(cost_pkl_path, "rb") as f:
            data = pickle.load(f)

        self.left_cost_table: Dict[str, float] = data["left_cost_table"]
        self.right_cost_table: Dict[str, float] = data["right_cost_table"]

        print(f"Loaded cost table: {len(self.left_cost_table)} left-reachable, "
              f"{len(self.right_cost_table)} right-reachable grid points.")

    # ----------------------------------------------------------------------- #
    # RealCost 任务采样                                                        #
    # ----------------------------------------------------------------------- #

    def sample_tasks_from_cost_table(
            self,
            n_per_window: Union[int, Dict[str, int]] = 5,
            seed: Optional[int] = None) -> pd.DataFrame:
        """
        从 cost table 网格点中采样任务（直接使用网格点，不做插值）。

        参数:
            n_per_window: 每个窗口的采样数量。
                - ``int``：所有窗口均采用相同数量
                - ``Dict[str, int]``：每个窗口独立指定，如
                  ``{"W_left_back": 6, "W_left_front": 4, ...}``
            seed: 随机种子（None 则不固定）

        返回:
            task_df，字段：region, x, y, z, table_key,
                           accessible_by, is_interference, time_to_L, time_to_R
        """
        rng = np.random.default_rng(seed)

        all_keys = set(self.left_cost_table.keys()) | set(self.right_cost_table.keys())
        task_data = []

        for window in self.windows:
            w_name = window["name"]
            n = (n_per_window if isinstance(n_per_window, int)
                 else n_per_window.get(w_name, 5))

            # 筛选落在该窗口内的可达网格点
            window_keys = [
                k for k in all_keys
                if (window["x_min"] <= float(k.split("_")[0]) <= window["x_max"] and
                    window["y_min"] <= float(k.split("_")[1]) <= window["y_max"])
            ]

            if not window_keys:
                print(f"  Warning: No reachable points in window '{w_name}', skipping.")
                continue

            n_actual = min(n, len(window_keys))
            sampled_keys = [
                str(k) for k in rng.choice(window_keys, size=n_actual, replace=False)
            ]

            for key in sampled_keys:
                parts = key.split("_")
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])

                l_cost = self.left_cost_table.get(key)
                r_cost = self.right_cost_table.get(key)

                accessible_by: List[str] = []
                if l_cost is not None:
                    accessible_by.append("L")
                if r_cost is not None:
                    accessible_by.append("R")

                if not accessible_by:
                    continue  # both_unreachable，跳过

                is_interference = (l_cost is not None and r_cost is not None)
                time_L = (2.0 * float(l_cost) + self.base_operation_time) if l_cost is not None else None
                time_R = (2.0 * float(r_cost) + self.base_operation_time) if r_cost is not None else None

                task_data.append({
                    "region":          w_name,
                    "x":               x,
                    "y":               y,
                    "z":               z,
                    "table_key":       key,
                    "accessible_by":   accessible_by,
                    "is_interference": is_interference,
                    "time_to_L":       time_L,
                    "time_to_R":       time_R,
                })

        if not task_data:
            raise RuntimeError("No tasks sampled. Check windows_config and cost table coverage.")

        self.task_df = pd.DataFrame(task_data).reset_index(drop=True)
        self._extract_milp_parameters()

        print(f"Sampled {len(self.task_df)} tasks across {len(self.windows)} windows:")
        for window in self.windows:
            wn = window["name"]
            n_w = int((self.task_df["region"] == wn).sum())
            n_int = int(
                ((self.task_df["region"] == wn) & self.task_df["is_interference"]).sum()
            )
            print(f"  {wn}: {n_w} tasks ({n_int} interference, {n_w - n_int} single-arm)")

        return self.task_df

    # ----------------------------------------------------------------------- #
    # RealCost 启发式                                                          #
    # ----------------------------------------------------------------------- #

    def spatial_order_heuristic(self) -> List[Dict]:
        """
        Real-cost 模式的简单贪心调度启发式。

        调度策略（三步顺序）：
          1. Left-only 任务：按 time_to_L 升序，依次排入左臂队列。
          2. Right-only 任务：按 time_to_R 升序，依次排入右臂队列
             （与第 1 步并行，两臂各自独立从 t=0 开始）。
          3. Both-reachable（干涉）任务：按 min(time_to_L, time_to_R) 升序，
             为每个任务选择能最早完成的手臂，同时遵守同窗口干涉任务串行约束。
        """
        if self.task_df is None:
            raise ValueError("No task data loaded. Call sample_tasks_from_cost_table() first.")

        actions: List[Dict] = []
        arm_free: Dict[str, float] = {"L": 0.0, "R": 0.0}

        # 每个窗口的最早可用时间（确保同窗口干涉任务串行）
        window_free: Dict[str, float] = {w["name"]: 0.0 for w in self.windows}

        # ---- 1) left-only 任务 ------------------------------------------- #
        left_only = self.task_df[
            self.task_df["accessible_by"].apply(
                lambda ab: "L" in ab and "R" not in ab
            )
        ].copy().sort_values("time_to_L")

        for idx, row in left_only.iterrows():
            start = arm_free["L"]
            end = start + float(row["time_to_L"])
            arm_free["L"] = end
            actions.append({
                "task": f"t{idx}", "arm": "L",
                "x": row["x"], "y": row["y"],
                "start": start, "end": end,
            })

        # ---- 2) right-only 任务 ------------------------------------------ #
        right_only = self.task_df[
            self.task_df["accessible_by"].apply(
                lambda ab: "R" in ab and "L" not in ab
            )
        ].copy().sort_values("time_to_R")

        for idx, row in right_only.iterrows():
            start = arm_free["R"]
            end = start + float(row["time_to_R"])
            arm_free["R"] = end
            actions.append({
                "task": f"t{idx}", "arm": "R",
                "x": row["x"], "y": row["y"],
                "start": start, "end": end,
            })

        # ---- 3) both-reachable（干涉）任务 --------------------------------- #
        both = self.task_df[self.task_df["is_interference"]].copy()
        both["_min_cost"] = both.apply(
            lambda r: min(
                float(r["time_to_L"]) if pd.notna(r["time_to_L"]) else float("inf"),
                float(r["time_to_R"]) if pd.notna(r["time_to_R"]) else float("inf"),
            ),
            axis=1,
        )
        both = both.sort_values("_min_cost")

        for idx, row in both.iterrows():
            region = row["region"]
            t_L = float(row["time_to_L"]) if pd.notna(row["time_to_L"]) else None
            t_R = float(row["time_to_R"]) if pd.notna(row["time_to_R"]) else None
            win_t = window_free.get(region, 0.0)

            # 选择能最早完成的手臂
            end_L = (max(arm_free["L"], win_t) + t_L) if t_L is not None else float("inf")
            end_R = (max(arm_free["R"], win_t) + t_R) if t_R is not None else float("inf")

            if end_L <= end_R:
                chosen = "L"
                start = max(arm_free["L"], win_t)
                end = end_L
                arm_free["L"] = end
            else:
                chosen = "R"
                start = max(arm_free["R"], win_t)
                end = end_R
                arm_free["R"] = end

            window_free[region] = end
            actions.append({
                "task": f"t{idx}", "arm": chosen,
                "x": row["x"], "y": row["y"],
                "start": start, "end": end,
            })

        return sorted(actions, key=lambda a: a["start"])

    # ----------------------------------------------------------------------- #
    # RealCost 绘图钩子                                                        #
    # ----------------------------------------------------------------------- #

    def _draw_environment_background(self, ax, with_region_labels: bool = True) -> None:
        """
        绘制采样窗口矩形（半透明填充）作为环境背景。

        参数:
            ax: matplotlib Axes 对象
            with_region_labels: True 时为窗口矩形添加图例标签。
        """
        for i, window in enumerate(self.windows):
            x0 = window["x_min"]
            y0 = window["y_min"]
            w = window["x_max"] - window["x_min"]
            h = window["y_max"] - window["y_min"]
            facecolor = self._WINDOW_FACECOLORS[i % len(self._WINDOW_FACECOLORS)]
            label = window["name"] if with_region_labels else "_nolegend_"
            ax.add_patch(plt.Rectangle(
                (x0, y0), w, h,
                facecolor=facecolor, alpha=0.25,
                edgecolor="gray", linewidth=1.5,
                label=label,
            ))

    def _get_task_color(self, task_id: str) -> str:
        """
        根据任务可达性返回绘图颜色：
        left-only → 蓝（steelblue），right-only → 红（tomato），both → 橙（darkorange）。
        """
        row = self.task_df.iloc[int(task_id[1:])]
        if row["is_interference"]:
            return self._ACCESSIBILITY_COLORS["both"]
        if "L" in row["accessible_by"]:
            return self._ACCESSIBILITY_COLORS["left_only"]
        return self._ACCESSIBILITY_COLORS["right_only"]

    # ----------------------------------------------------------------------- #
    # RealCost 配置保存                                                        #
    # ----------------------------------------------------------------------- #

    def _save_configuration(self, config_dir: str) -> None:
        """保存 real-cost 模式实验配置（JSON）。"""
        import json

        config = {
            "mode": "real_cost",
            "cost_pkl_path": str(self.cost_pkl_path),
            "arm_bases": {
                "left":  self.L_base.tolist(),
                "right": self.R_base.tolist(),
            },
            "windows": self.windows,
            "total_tasks": len(self.task_df) if self.task_df is not None else 0,
            "accessibility_colors": self._ACCESSIBILITY_COLORS,
        }

        with open(os.path.join(config_dir, "experiment_config.json"), "w") as f:
            json.dump(config, f, indent=2)


# ===========================================================================
# 向后兼容别名
# DualArmPlanner 指向 BaselinePlanner，确保现有调用代码（如 baseline.py）无需修改
# ===========================================================================

DualArmPlanner = BaselinePlanner
