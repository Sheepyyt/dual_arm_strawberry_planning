import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import os
import pickle
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Union


# =========================================================
# 默认区域配置（B1~B6 正方形区域），两种模式共用
# B2/B5 为干涉区，B1/B3/B4/B6 为非干涉区（各臂专属区）
# =========================================================
_DEFAULT_REGIONS = [
    {"center": (-0.4,  0.5), "width": 0.4, "height": 0.4, "name": "B1", "arm_access": ["L"]},
    {"center": ( 0.0,  0.5), "width": 0.4, "height": 0.4, "name": "B2", "arm_access": ["L", "R"]},
    {"center": ( 0.4,  0.5), "width": 0.4, "height": 0.4, "name": "B3", "arm_access": ["R"]},
    {"center": (-0.4, -0.5), "width": 0.4, "height": 0.4, "name": "B4", "arm_access": ["L"]},
    {"center": ( 0.0, -0.5), "width": 0.4, "height": 0.4, "name": "B5", "arm_access": ["L", "R"]},
    {"center": ( 0.4, -0.5), "width": 0.4, "height": 0.4, "name": "B6", "arm_access": ["R"]},
]
_DEFAULT_COLORS = ["red", "orange", "green", "blue", "purple", "brown"]
_DEFAULT_INTERFERENCE = ["B2", "B5"]


# =========================================================
# 抽象基类：DualArmPlannerCore
# 包含两种模式完全共用的所有模块：
#   热启动、启发式算法、MILP 建模、解提取、
#   结果保存、动画、甘特图等可视化函数。
# 子类只需实现：
#   create_task_dataset() —— 采样方式（连续 vs 离散网格）
#   _compute_processing_time() —— 处理时间来源（欧式距离 vs cost table）
# =========================================================
class DualArmPlannerCore(ABC):
    
    def __init__(self,
                 L_base: np.ndarray,
                 R_base: np.ndarray,
                 regions_config: List[Dict],
                 colors: List[str],
                 interference_regions: List[str],
                 base_operation_time: float = 0.0):
        """
        参数:
            L_base: 左臂基座位置 (x, y)
            R_base: 右臂基座位置 (x, y)
            regions_config: 采摘区域配置列表
            colors: 各区域对应颜色（与 regions_config 等长）
            interference_regions: 干涉区域名称列表
            base_operation_time: 每颗草莓的固定处理时间
        """
        self.L_base = L_base
        self.R_base = R_base
        self.regions = regions_config
        self.colors = colors
        self.interference_regions = interference_regions
        self.base_operation_time = base_operation_time

        # 数据存储
        self.task_df = None
        self.milp_params = None
        self.heuristic_actions = None
        self.milp_actions = None
        self.improvement = 0

    # ------------------------------------------------------------------
    # 子类必须实现的抽象方法
    # ------------------------------------------------------------------

    @abstractmethod
    def create_task_dataset(self, points_per_region: Dict[str, int]) -> pd.DataFrame:
        """
        创建任务数据集。
        Baseline: 在正方形区域内连续随机采样
        RealCost: 从 cost table 离散网格点中采样
        """
        pass

    @abstractmethod
    def _compute_processing_time(self, point: np.ndarray, arm: str) -> Optional[float]:
        """
        计算单颗草莓对某只臂的处理时间（秒）。
        返回 None 表示该臂不可达此点。
        Baseline: 2 * 欧式距离 + base_operation_time
        RealCost: 2 * cost_table 单程代价 + base_operation_time
        """
        pass

    # 在指定矩形范围内生成随机点
    def generate_random_points(self, center: Tuple[float, float], width: float, height: float, n: int = 5) -> np.ndarray:
        cx, cy = center
        xs = np.random.uniform(cx - width / 2, cx + width / 2, n)
        ys = np.random.uniform(cy - height / 2, cy + height / 2, n)
        return np.vstack((xs, ys)).T

    # 读入指定任务数据集
    def load_task_locations(self, task_locations: List[Dict]) -> pd.DataFrame:
        """
        从字典列表中加载任务位置。
        处理时间由子类的 _compute_processing_time() 计算。
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
            
            pt = np.array([x, y])
            time_L = self._compute_processing_time(pt, "L") if "L" in arm_access else None
            time_R = self._compute_processing_time(pt, "R") if "R" in arm_access else None
            
            task_data.append({
                "region": region_name,
                "x": x,
                "y": y,
                "accessible_by": list(arm_access),
                "time_to_L": time_L,
                "time_to_R": time_R,
            })
        
        self.task_df = pd.DataFrame(task_data)
        self._extract_milp_parameters()
        return self.task_df

    # 从 task_df 中提取 MILP 所需的参数
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

    # 求解目标函数的优化问题
    def solve_optimization(self, time_limit: int = 30, 
                           heuristic_name: str = "spatial_order") -> Tuple[List[Dict], List[Dict], float]:
        """
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

    # 空间顺序启发式算法
    def spatial_order_heuristic(self) -> List[Dict]:
        """
        基于区域的空间顺序启发式算法（两种模式共用）。
        左臂: B2 → B1 → B4 (先干涉区域，然后是自己的区域)
        右臂: B5 → B6 → B3 (先干涉区域，然后是自己的区域)
        最后对任何未被调度的任务做兜底处理。
        """
        if self.task_df is None:
            raise ValueError("No task data loaded.")
        
        actions = []
        arm_times = {'L': 0.0, 'R': 0.0}
        region_busy_until = {r: 0.0 for r in self.interference_regions}
        
        # 更新序列：每个手臂优先处理其干涉区域
        left_arm_order = ['B2', 'B1', 'B4']   # 左臂：干涉区 B2，然后是仅左臂区域
        right_arm_order = ['B5', 'B6', 'B3']  # 右臂：干涉区 B5，然后是仅右臂区域 
                
        # 处理单个区域的核心调度器：负责规划某一只手臂在某一个具体区域内的所有动作
        def schedule_region_tasks(arm, region, start_time):
            region_tasks = self.task_df[
                (self.task_df['region'] == region) & 
                (self.task_df['accessible_by'].apply(lambda x: arm in x))
            ].copy()
            
            if region_tasks.empty:
                return [], start_time
            
            # 获取当前手臂摘每颗草莓的预估时间，并按升序排列（贪心策略：先采最快的）
            region_tasks["proc_time"] = region_tasks[f"time_to_{arm}"]
            region_tasks = region_tasks.sort_values("proc_time")
            current_time = start_time
            
            # 调度该区域内的所有任务
            region_actions = []
            for idx, row in region_tasks.iterrows():
                if region in self.interference_regions:
                    # 如果是干涉区，必须等上一个占用的任务释放该区域
                    actual_start = max(current_time, region_busy_until[region])
                    region_busy_until[region] = actual_start + row[f"time_to_{arm}"]
                else:
                    # 非干涉区，到场即可开工
                    actual_start = current_time
                end_time = actual_start + row[f"time_to_{arm}"]
                
                # 记录该任务的动作信息
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
        
        # 兜底：处理未被以上两轮调度到的任务
        # （例如干涉区内仅对另一臂可达的草莓，或区域名不在默认顺序中的草莓）
        scheduled_tasks = {a["task"] for a in actions}
        for idx, row in self.task_df.iterrows():
            task_id = f"t{idx}"
            if task_id not in scheduled_tasks:
                arm = row["accessible_by"][0]
                region = row["region"]
                if region in self.interference_regions:
                    actual_start = max(arm_times[arm], region_busy_until.get(region, 0.0))
                    region_busy_until[region] = actual_start + row[f"time_to_{arm}"]
                else:
                    actual_start = arm_times[arm]
                end_time = actual_start + row[f"time_to_{arm}"]
                actions.append({
                    "task": task_id,
                    "arm": arm,
                    "x": row["x"],
                    "y": row["y"],
                    "start": actual_start,
                    "end": end_time,
                })
                arm_times[arm] = end_time

        return sorted(actions, key=lambda x: x["start"])

    # 把 MILP 约束翻译成 Gurobi 能理解的形式
    def build_milp_model(self, warm_start_actions: Optional[List[Dict]] = None) -> gp.Model:
        """
        构建目标函数的 MILP 模型
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
        
        # 约束条件
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
        
        # 约束 4. 干涉区内的任务必须串行（按区域分组，动态处理，不硬编码区域名）
        interference_groups: Dict[str, List[str]] = defaultdict(list)
        for task_id in interference_tasks:
            interference_groups[self.milp_params["regions"][task_id]].append(task_id)
        
        for group_tasks in interference_groups.values():
            for idx_i in range(len(group_tasks)):
                for idx_j in range(idx_i + 1, len(group_tasks)):
                    i = group_tasks[idx_i]
                    j = group_tasks[idx_j]
                    
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

    # 从求解的模型中提取解变量
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

    # 从解变量中提取动作序列
    def _extract_action_sequence(self, x_vals: Dict, t_vals: Dict) -> List[Dict]:
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
    
    # 根据启发式动作设置热启动值
    def _set_warm_start(self, model: gp.Model, actions: List[Dict]):
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

            # ---- 4) region_order 变量：同理，先默认，再用启发式干涉区开始时间覆盖 ----
            for (i, j), v in region_order_vars.items():
                if task_index.get(i, 0) < task_index.get(j, 0):
                    v.Start = 1
                else:
                    v.Start = 0

            interference_tasks = self.milp_params["interference_set"]
            interference_groups: Dict[str, List[str]] = defaultdict(list)
            for tid in interference_tasks:
                interference_groups[self.milp_params["regions"][tid]].append(tid)

            for group_tids in interference_groups.values():
                task_times = [(tid, t_warm[tid]) for tid in group_tids if tid in t_warm]
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
    
    # 保存结果
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

    # 创建采摘解决方案的 gif 动画
    def animate_solution(self, actions: List[Dict], save_path: Optional[str] = None, 
                        title: str = "Dual-Arm Harvesting Animation") -> animation.FuncAnimation:
        """
        创建采摘解决方案的动画
        参数:
            actions: 动作字典列表
            save_path: 将动画保存为 GIF 的可选路径
            title: 动画标题
        返回:
            动画对象
        """
        if self.task_df is None:
            raise ValueError("No task data loaded.")
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 绘制区域
        for i, region in enumerate(self.regions):
            cx, cy = region["center"]
            w, h = region["width"], region["height"]
            name = region["name"]
            color = self.colors[i]
            ax.add_patch(plt.Rectangle((cx - w/2, cy - h/2), w, h, 
                                     fill=False, edgecolor=color, linewidth=2, label=name))
        
        # 绘制手臂基座
        arm_L, = ax.plot(*self.L_base, 'ks', markersize=10, label='Left Arm Base (L)')
        arm_R, = ax.plot(*self.R_base, 'ko', markersize=10, label='Right Arm Base (R)')
        
        # 绘制操作边界
        ax.add_patch(plt.Rectangle((-0.5, -0.25), 1.0, 0.5, fill=False, 
                                 edgecolor='black', linestyle='--', linewidth=1.5, 
                                 label='Operation Boundary'))
        
        # 设置绘图
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_title(title)

        # 按区域名建立配色索引（两种模式通用）
        region_color_map = {r["name"]: self.colors[i] for i, r in enumerate(self.regions)}
        
        # 初始化任务点
        task_points = {}
        task_states = {}
        
        for act in actions:
            task_id = act['task']
            region_name = self.task_df.iloc[int(task_id[1:])]['region']
            color = region_color_map.get(region_name, 'gray')
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
                    
                    region_name = self.task_df.iloc[int(task_id[1:])]['region']
                    color = region_color_map.get(region_name, 'gray')
                    
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
    
    # 画甘特图
    def plot_gantt_chart(self, actions: List[Dict], save_path: Optional[str] = None, 
                         title_suffix: str = "") -> None:
        """
        创建甘特图可视化
        参数:
            actions: 动作字典列表
            save_path: 将图表保存为 PNG 的可选路径
            title_suffix: 标题的附加文本
        """
        if self.task_df is None:
            raise ValueError("No task data loaded.")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 动态生成区域配色映射（两种模式通用）
        region_colors = {r["name"]: self.colors[i] for i, r in enumerate(self.regions)}
        
        left_actions = [a for a in actions if a['arm'] == 'L']
        right_actions = [a for a in actions if a['arm'] == 'R']
        
        # 绘制任务
        for action in left_actions:
            task_id = action['task']
            task_idx = int(task_id[1:])
            region = self.task_df.iloc[task_idx]['region']
            color = region_colors.get(region, 'gray')
            
            ax.barh(0, action['end'] - action['start'], left=action['start'], 
                   height=0.4, color=color, alpha=0.7, edgecolor='black')
            
            mid_time = (action['start'] + action['end']) / 2
            ax.text(mid_time, 0, f"{task_id}\n{region}", ha='center', va='center', 
                   fontsize=8, fontweight='bold')
        
        for action in right_actions:
            task_id = action['task']
            task_idx = int(task_id[1:])
            region = self.task_df.iloc[task_idx]['region']
            color = region_colors.get(region, 'gray')
            
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
        
        # 添加图例和完工时间
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.7, 
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
    

    # 保存综合结果
    def save_comprehensive_results(self, result_dir: str = "comprehensive_results", 
                                 experiment_name: str = None, 
                                 include_data: bool = True,
                                 include_config: bool = True) -> None:
        """
        使用详细分析和有组织的文件夹结构保存综合优化结果
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
            import time
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
        print("  → Generating animations...")
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
        print("  → Creating Gantt charts...")
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
        print("  → Creating task distribution plot...")
        self._plot_task_distribution(os.path.join(folders['plots'], 'task_distribution.png'))
        
        # 4. 保存区域利用率分析
        print("  → Creating region utilization analysis...")
        self._plot_region_utilization(folders['plots'])
        
        # 5. 保存原始数据（如果请求）
        if include_data:
            print("  → Saving raw data...")
            self._save_raw_data(folders['data'])
        
        # 6. 保存配置（如果请求）
        if include_config:
            print("  → Saving configuration...")
            self._save_configuration(folders['config'])
        
        # 7. 保存综合摘要
        print("  → Creating comprehensive summary...")
        self._save_comprehensive_summary(base_dir, heuristic_makespan, milp_makespan)
        
        # 8. 保存性能比较
        if self.milp_actions is not None:
            print("  → Creating performance comparison...")
            self._create_performance_comparison(folders['analysis'])
        
        print(f"✓ Comprehensive results saved successfully!")
        print(f"  Main directory: {base_dir}")
        print(f"  Animations: {folders['animations']}")
        print(f"  Plots: {folders['plots']}")
        print(f"  Analysis: {folders['analysis']}")
        
        # 关闭所有 matplotlib 图形以防止内存问题
        plt.close('all')
    
    # 创建显示各区域任务分布的图
    def _plot_task_distribution(self, save_path: str) -> None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 按区域统计任务数量
        region_counts = self.task_df['region'].value_counts().sort_index()
        region_color_map = {r["name"]: self.colors[i] for i, r in enumerate(self.regions)}
        colors = [region_color_map.get(name, 'gray') for name in region_counts.index]
        
        ax1.bar(region_counts.index, region_counts.values, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_title('Task Count by Region')
        ax1.set_xlabel('Region')
        ax1.set_ylabel('Number of Tasks')
        ax1.grid(True, alpha=0.3)
        
        # 任务位置散点图
        for i, region in enumerate(self.regions):
            region_tasks = self.task_df[self.task_df['region'] == region['name']]
            if not region_tasks.empty:
                ax2.scatter(region_tasks['x'], region_tasks['y'], 
                           color=self.colors[i], label=region['name'], s=50, alpha=0.7)
        
        # 绘制区域边界
        for i, region in enumerate(self.regions):
            cx, cy = region["center"]
            w, h = region["width"], region["height"]
            ax2.add_patch(plt.Rectangle((cx - w/2, cy - h/2), w, h, 
                                       fill=False, edgecolor=self.colors[i], linewidth=2))
        
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
    
    # 创建区域利用率分析图
    def _plot_region_utilization(self, plots_dir: str) -> None:
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
            
            # 绘制条形图
            regions = [r["name"] for r in self.regions]
            left_times = [region_times["L"].get(region, 0) for region in regions]
            right_times = [region_times["R"].get(region, 0) for region in regions]
            
            x = np.arange(len(regions))
            width = 0.35
            
            ax.bar(x - width/2, left_times, width, label='Left Arm', color='blue', alpha=0.7)
            ax.bar(x + width/2, right_times, width, label='Right Arm', color='red', alpha=0.7)
            
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
            
            ax3.bar(x - width/2, left_workloads, width, label='Left Arm', color='blue', alpha=0.7)
            ax3.bar(x + width/2, right_workloads, width, label='Right Arm', color='red', alpha=0.7)
            ax3.set_xlabel('Solution Method')
            ax3.set_ylabel('Total Workload (s)')
            ax3.set_title('Arm Workload Comparison')
            ax3.set_xticks(x)
            ax3.set_xticklabels(methods)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 添加值标签
            for i, (left, right) in enumerate(zip(left_workloads, right_workloads)):
                ax3.text(i - width/2, left + 0.1, f'{left:.1f}', ha='center', va='bottom')
                ax3.text(i + width/2, right + 0.1, f'{right:.1f}', ha='center', va='bottom')
        
        # 时间线可视化
        if self.milp_actions is not None:
            self._plot_arm_timeline(ax4)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'region_utilization.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

    # 绘制手臂时间线图
    def _plot_arm_timeline(self, ax) -> None:
        actions = self.milp_actions if self.milp_actions else self.heuristic_actions
        
        left_actions = [a for a in actions if a['arm'] == 'L']
        right_actions = [a for a in actions if a['arm'] == 'R']
        
        # 绘制时间线图
        for i, action in enumerate(left_actions):
            ax.barh(0, action['end'] - action['start'], left=action['start'], 
                   height=0.3, color='blue', alpha=0.7)
        
        for i, action in enumerate(right_actions):
            ax.barh(1, action['end'] - action['start'], left=action['start'], 
                   height=0.3, color='red', alpha=0.7)
        
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Left Arm', 'Right Arm'])
        ax.set_xlabel('Time (s)')
        ax.set_title('Arm Activity Timeline')
        ax.grid(True, alpha=0.3)
    
    # 保存原始数据文件
    def _save_raw_data(self, data_dir: str) -> None:
        # 保存任务数据
        self.task_df.to_csv(os.path.join(data_dir, 'task_data.csv'), index=False)
        
        # 保存动作序列
        if self.heuristic_actions:
            heuristic_df = pd.DataFrame(self.heuristic_actions)
            heuristic_df.to_csv(os.path.join(data_dir, 'heuristic_actions.csv'), index=False)
        
        if self.milp_actions:
            milp_df = pd.DataFrame(self.milp_actions)
            milp_df.to_csv(os.path.join(data_dir, 'milp_actions.csv'), index=False)

    # 保存配置和参数
    def _save_configuration(self, config_dir: str) -> None:
        import json
        
        config = {
            "arm_bases": {
                "left": self.L_base.tolist(),
                "right": self.R_base.tolist()
            },
            "regions": self.regions,
            "interference_regions": self.interference_regions,
            "total_tasks": len(self.task_df) if self.task_df is not None else 0,
            "region_colors": self.colors
        }
        
        with open(os.path.join(config_dir, 'experiment_config.json'), 'w') as f:
            json.dump(config, f, indent=2)
    
    # 保存综合分析摘要
    def _save_comprehensive_summary(self, base_dir: str, heuristic_makespan: float, 
                                  milp_makespan: float) -> None:
        summary_path = os.path.join(base_dir, 'EXPERIMENT_SUMMARY.md')
        
        with open(summary_path, 'w') as f:
            f.write("# Dual-Arm Harvesting Optimization Results\n\n")
            f.write(f"**Experiment Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Problem Configuration\n")
            f.write(f"- **Total Tasks**: {len(self.task_df)}\n")
            f.write(f"- **Regions**: {', '.join([r['name'] for r in self.regions])}\n")
            f.write(f"- **Interference Regions**: {', '.join(self.interference_regions)}\n")
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
        
        print(f"  ✓ Summary saved to: EXPERIMENT_SUMMARY.md")
    
    # 创建性能比较图表
    def _create_performance_comparison(self, analysis_dir: str) -> None:
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
        
        # Add value labels
        for bar, value in zip(bars1, makespans):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{value:.2f}s', ha='center', va='bottom', fontweight='bold')
        
        # Add improvement percentage
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

# =========================================================
# BaselinePlanner：Baseline 模式子类
# 欧式距离近似处理时间 + 连续随机采样
# =========================================================
class BaselinePlanner(DualArmPlannerCore):
    """
    Baseline 模式：B1~B6 正方形区域 + 欧式距离处理时间 + 连续随机采样。
    与原有 DualArmPlanner 行为完全一致。
    所有核心模块（热启动、启发式、MILP、可视化）从 DualArmPlannerCore 继承。
    """

    def __init__(self,
                 L_base: np.ndarray = np.array([-0.3, 0.]),
                 R_base: np.ndarray = np.array([0.3, 0.]),
                 regions_config: Optional[List[Dict]] = None,
                 base_operation_time: float = 0.0):
        super().__init__(
            L_base=L_base,
            R_base=R_base,
            regions_config=regions_config if regions_config is not None else _DEFAULT_REGIONS,
            colors=_DEFAULT_COLORS,
            interference_regions=_DEFAULT_INTERFERENCE,
            base_operation_time=base_operation_time,
        )

    def _compute_processing_time(self, point: np.ndarray, arm: str) -> float:
        """处理时间 = 2 * 欧式距离（单程近似）+ 固定处理时间。"""
        base = self.L_base if arm == "L" else self.R_base
        one_way_time = float(np.linalg.norm(point - base))
        return 2.0 * one_way_time + self.base_operation_time

    def create_task_dataset(self, points_per_region: Dict[str, int]) -> pd.DataFrame:
        """
        在每个正方形区域内连续随机采样，处理时间用欧式距离近似。
        accessible_by 直接取自区域配置（arm_access）。
        """
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
                    "accessible_by": list(arm_access),
                    "time_to_L": time_L,
                    "time_to_R": time_R,
                })
        self.task_df = pd.DataFrame(task_data)
        self._extract_milp_parameters()
        return self.task_df


# =========================================================
# RealCostPlanner：Real Cost 模式子类
# cost table 真实 IK 代价 + cost table 离散网格采样
# =========================================================
class RealCostPlanner(DualArmPlannerCore):
    """
    Real Cost 模式：保留 B1~B6 正方形区域（保守安全几何假设），
    用 cost table 中的真实 IK 代价替代欧式距离，
    任务点在 cost table 离散网格中采样（不做插值）。

    非干涉区 (B1/B3/B4/B6)：强制只允许主负责臂（保守安全策略）。
    干涉区 (B2/B5)：根据 cost table 实际可达性决定 accessible_by（方案 A）。
    所有核心模块（热启动、启发式、MILP、可视化）从 DualArmPlannerCore 继承。
    """

    # 非干涉区与其主负责臂的映射（保守安全策略）
    _REGION_PRIMARY_ARM: Dict[str, str] = {"B1": "L", "B4": "L", "B3": "R", "B6": "R"}

    def __init__(self,
                 cost_table_path: str,
                 L_base: np.ndarray = np.array([-0.3, 0.]),
                 R_base: np.ndarray = np.array([0.3, 0.]),
                 regions_config: Optional[List[Dict]] = None,
                 base_operation_time: float = 0.0,
                 step_xyz: float = 0.02,
                 z: float = 0.56):
        """
        参数:
            cost_table_path: dual_arm_cost.pkl 文件路径
            step_xyz: cost table 的空间网格步长（与 build_roi_table.py 一致）
            z: 草莓所在高度（vehicle frame，与 cost table 扫描高度一致）
        """
        super().__init__(
            L_base=L_base,
            R_base=R_base,
            regions_config=regions_config if regions_config is not None else _DEFAULT_REGIONS,
            colors=_DEFAULT_COLORS,
            interference_regions=_DEFAULT_INTERFERENCE,
            base_operation_time=base_operation_time,
        )
        self._step_xyz = step_xyz
        self._z = z

        with open(cost_table_path, "rb") as f:
            cost_data = pickle.load(f)
        self.left_cost_table: Dict[str, float] = cost_data["left_cost_table"]
        self.right_cost_table: Dict[str, float] = cost_data["right_cost_table"]

    def _snap_key(self, x: float, y: float, z: float) -> str:
        """将坐标对齐到 cost table 网格步长并生成查询 key。"""
        s = self._step_xyz
        xs = round(round(x / s) * s, 3)
        ys = round(round(y / s) * s, 3)
        zs = round(round(z / s) * s, 3)
        return f"{xs:.3f}_{ys:.3f}_{zs:.3f}"

    def _lookup_cost(self, point_xy: np.ndarray, arm: str) -> Optional[float]:
        """
        从 cost table 查找单程运动代价。
        返回 None 表示该臂不可达此点。直接用网格 key 查值，不做插值。
        """
        key = self._snap_key(float(point_xy[0]), float(point_xy[1]), self._z)
        table = self.left_cost_table if arm == "L" else self.right_cost_table
        return table.get(key, None)

    def _compute_processing_time(self, point: np.ndarray, arm: str) -> Optional[float]:
        """处理时间 = 2 * cost_table 单程代价 + 固定处理时间。返回 None 表示该臂不可达。"""
        one_way_cost = self._lookup_cost(point, arm)
        if one_way_cost is None:
            return None
        return 2.0 * one_way_cost + self.base_operation_time

    def create_task_dataset(self, points_per_region: Dict[str, int]) -> pd.DataFrame:
        """
        在 cost table 离散网格点中采样任务（不做插值）：
          - 遍历 cost table 的 key，筛选落在各正方形区域内、z 匹配的 key
          - 从候选 key 中随机抽取 num_points 个（不放回）
          - 非干涉区：accessible_by 强制为主负责臂（保守安全）
          - 干涉区：accessible_by 取决于 cost table 实际可达性（方案 A）
        """
        # 合并左右表的所有 key，覆盖两臂各自的可达范围
        all_keys = set(self.left_cost_table.keys()) | set(self.right_cost_table.keys())
        half_step = self._step_xyz / 2.0

        task_data = []
        for region in self.regions:
            cx, cy = region["center"]
            w, h = region["width"], region["height"]
            name = region["name"]
            num_points = points_per_region.get(name, 5)

            x_lo, x_hi = cx - w / 2, cx + w / 2
            y_lo, y_hi = cy - h / 2, cy + h / 2

            # 筛选在本区域内且 z 匹配的 grid 点
            candidate_keys = []
            for key in all_keys:
                parts = key.split("_")
                x_k, y_k, z_k = float(parts[0]), float(parts[1]), float(parts[2])
                if (x_lo <= x_k <= x_hi and y_lo <= y_k <= y_hi
                        and abs(z_k - self._z) < half_step):
                    candidate_keys.append(key)

            if not candidate_keys:
                print(f"Warning: No cost table grid points found in region {name}")
                continue

            n_sample = min(num_points, len(candidate_keys))
            chosen_indices = np.random.choice(len(candidate_keys), size=n_sample, replace=False)

            is_interference = name in self.interference_regions
            primary_arm = self._REGION_PRIMARY_ARM.get(name)  # None for interference regions

            for idx in chosen_indices:
                key = candidate_keys[idx]
                parts = key.split("_")
                x_k, y_k = float(parts[0]), float(parts[1])
                pt = np.array([x_k, y_k])

                if not is_interference:
                    # 非干涉区：强制只允许主负责臂（保守安全策略）
                    t_primary = self._compute_processing_time(pt, primary_arm)
                    if t_primary is None:
                        continue  # 主臂不可达此点，跳过
                    accessible_by = [primary_arm]
                    time_L = t_primary if primary_arm == "L" else None
                    time_R = t_primary if primary_arm == "R" else None
                else:
                    # 干涉区：使用 cost table 真实可达性（方案 A）
                    time_L = self._compute_processing_time(pt, "L")
                    time_R = self._compute_processing_time(pt, "R")
                    accessible_by = []
                    if time_L is not None:
                        accessible_by.append("L")
                    if time_R is not None:
                        accessible_by.append("R")
                    if not accessible_by:
                        continue  # 双臂均不可达，跳过

                task_data.append({
                    "region": name,
                    "x": float(x_k),
                    "y": float(y_k),
                    "accessible_by": accessible_by,
                    "time_to_L": time_L,
                    "time_to_R": time_R,
                })

        self.task_df = pd.DataFrame(task_data)
        self._extract_milp_parameters()
        return self.task_df


# =========================================================
# 向后兼容别名
# =========================================================
# DualArmPlanner 原先指向单一类，现在等同于 BaselinePlanner。
# 已有脚本中的 from dual_arm_planner import DualArmPlanner 无需修改。
DualArmPlanner = BaselinePlanner
