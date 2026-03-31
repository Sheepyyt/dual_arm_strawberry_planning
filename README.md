# 在固定停车点下的双机械臂草莓采摘规划

## 项目概述

本项目研究双机械臂车辆在**固定停车点**下的最优草莓采摘顺序规划，目标是**最小化总完工时间（Makespan）**。车辆停在固定位置，两条机械臂分别从各自基座出发对周围草莓区域执行采摘任务，完成每次采摘后需回到初始位置（篮子）放置草莓，再执行下一次采摘。

---

## 问题定义

### 核心假设与设定

1. **采摘-放置循环（Pick-and-Place Cycle）**  
   每次采摘完一颗草莓后，机械臂需要将其放入车上的篮子。因此每次采摘动作均从"基座（篮子位置）→ 草莓 → 基座"构成一个完整的往返动作。每颗草莓的**处理时间**定义为：
   ```
   处理时间 = 2 × 单程移动时间 + 固定处理时间
   ```
   其中固定处理时间（`base_operation_time`）包含夹取、放置等操作耗时。

2. **区域划分：干涉区与非干涉区**  
   为简化三维碰撞检测问题，将草莓生长空间划分为 6 个正方形区域（B1～B6）：

   | 区域 | 位置     | 可达臂 | 类型       |
   |------|----------|--------|------------|
   | B1   | 左上     | 左臂   | 非干涉区   |
   | B2   | 中上     | 双臂均可达 | **干涉区** |
   | B3   | 右上     | 右臂   | 非干涉区   |
   | B4   | 左下     | 左臂   | 非干涉区   |
   | B5   | 中下     | 双臂均可达 | **干涉区** |
   | B6   | 右下     | 右臂   | 非干涉区   |

   - **非干涉区**（B1/B3/B4/B6）：双臂可以同时在此区域独立作业，不会发生碰撞。
   - **干涉区**（B2/B5）：双臂不得同时在该区域内作业，必须**串行执行**，即一臂在干涉区采摘时，另一臂必须等待或在其他区域作业。

3. **末端执行器姿态约束**  
   末端仅可绕 yaw 方向转动（roll/pitch 固定为名义值）。采摘时对每个目标点沿 yaw 方向进行离散搜索，选择 IK 可解且代价最优的姿态。

4. **平面化处理（当前实现）**  
   当前代码仅考虑二维平面规划（z = 0.56m 固定高度切片），忽略垂直方向变化。

---

## 两种规划模式

本项目提供两种规划模式，均支持启发式算法和 MILP 精确优化，可按需选择：

---

### 模式一：Baseline 模式（`BaselinePlanner`）

**适用场景**：快速原型验证、算法对比实验、无需 IK 求解的轻量场景。

**机械臂模型来源**：手工设定基座坐标。
- 默认左臂基座：`[-0.3, 0.0]`
- 默认右臂基座：`[0.3, 0.0]`

**单程移动代价**：用欧氏距离近似移动时间（无需 IK 计算）。

**任务生成方式**：在矩形区域内连续均匀随机采样，坐标可以是区域内任意浮点数。

**启发式调度顺序**：
- 左臂：B2 → B1 → B4（先处理干涉区，再处理本侧区域）
- 右臂：B5 → B6 → B3

**MILP 可达性**：直接使用区域配置的 `arm_access` 字段。

**代码示例**：
```python
from planning.dual_arm_planner import BaselinePlanner  # 或 DualArmPlanner（向后兼容别名）
import numpy as np

np.random.seed(42)
planner = BaselinePlanner()
planner.create_task_dataset({"B1": 5, "B2": 2, "B3": 2, "B4": 3, "B5": 2, "B6": 4})
heuristic, milp, improvement = planner.solve_optimization(time_limit=30)
planner.save_results("results/baseline_run")
```

参见 `planning/baseline.py` 获取完整使用示例。

---

### 模式二：Real Cost 模式（`RealCostPlanner`）

**适用场景**：追求更真实代价模型的规划实验，需预先生成 cost table。

**机械臂模型来源**：从 `urdf/dual_arm_ik_xy_centered.urdf` 文件解析，与 `roi/build_roi_table.py` 中生成 cost table 时完全一致，确保代价模型与调度模型之间的一致性。
- 左臂基座（x, y）：由 URDF 中 `vehicle_to_left_arm` 关节解析得到
- 右臂基座（x, y）：由 URDF 中 `vehicle_to_right_arm` 关节解析得到

**单程移动代价**：从预计算的 cost table（`roi/results/dual_arm_cost.pkl`）查表获得，代价值为从 home 姿态出发的最小关节空间运动量（IK 真实代价），不使用欧氏距离近似。

**任务生成方式**：只从 cost table 中的可达网格离散点（步长 0.02m）里采样，不支持连续坐标或插值。

**区域可达性策略**：
- **B1/B4（强制 L-only）**：即使 cost table 显示右臂也可达，也仍强制设为仅左臂可达，`accessible_by = ["L"]`。
- **B3/B6（强制 R-only）**：即使 cost table 显示左臂也可达，也仍强制设为仅右臂可达，`accessible_by = ["R"]`。
- **B2/B5（干涉区，按点可达性）**：
  - 某点仅左臂可达 → `accessible_by = ["L"]`
  - 某点仅右臂可达 → `accessible_by = ["R"]`
  - 某点双臂均可达 → `accessible_by = ["L", "R"]`

**启发式调度**（预分配 + 分离调度）：
1. 预分配 B2/B5 中每颗草莓的归属：
   - B2：仅左臂可达的点和双臂均可达的点均分配给左臂，仅右臂可达的点分配给右臂
   - B5：仅右臂可达的点和双臂均可达的点均分配给右臂，仅左臂可达的点分配给左臂
2. 按预分配结果调度（左右臂同时从 t=0 开始工作）：
   - 左臂：B2（L 分配任务）→ B1 → B4 → B5（L 分配任务）
   - 右臂：B5（R 分配任务）→ B6 → B3 → B2（R 分配任务）

**代码示例**：
```python
from planning.dual_arm_planner import RealCostPlanner
import numpy as np

np.random.seed(42)
planner = RealCostPlanner()  # 自动从默认路径加载 URDF 和 cost table
planner.create_task_dataset({"B1": 5, "B2": 3, "B3": 5, "B4": 4, "B5": 3, "B6": 4})
heuristic, milp, improvement = planner.solve_optimization(time_limit=30)
planner.save_results("results/real_cost_run")
```

参见 `planning/real_cost_example.py` 获取完整使用示例。

---

## 优化算法

两种模式共用相同的 MILP 建模与启发式框架：

### 启发式算法（Spatial Order Heuristic）
- 基于区域空间顺序的贪心调度，优先处理干涉区，再处理本侧专属区域。
- 在干涉区内，按处理时间升序调度，并尊重区域互斥约束。
- 产生可行初解，作为 MILP 求解器的热启动（Warm Start）。

### MILP 精确优化
- 基于 Gurobi 求解器，构建混合整数线性规划模型。
- 决策变量包括：任务分配变量（`x`）、任务开始时间（`t`）、完工时间（`T`）、顺序变量（`o`、`y`）。
- 约束包括：分配唯一性、可达性、同臂顺序、干涉区串行、完工时间下界。
- 目标：最小化总完工时间 T。

详见 `planning/milp_formulation.md` 获取完整数学建模说明。

---

## 类继承结构

```
DualArmPlannerCore（基类）
├── BaselinePlanner（Baseline 模式）
│     别名：DualArmPlanner（向后兼容）
└── RealCostPlanner（Real Cost 模式）
```

- `DualArmPlannerCore`：封装所有共用逻辑（MILP 建模、求解、热启动、可视化、结果保存等）。
- `BaselinePlanner`：覆写代价函数（欧氏距离）、任务采样（连续随机）和启发式（B2→B1→B4）。
- `RealCostPlanner`：覆写代价函数（cost table 查表）、任务采样（网格点）、启发式（预分配版）；机械臂定义完全来自 URDF。

---

## 目录结构

```
.
├── planning/
│   ├── dual_arm_planner.py      # 核心规划器（DualArmPlannerCore / BaselinePlanner / RealCostPlanner）
│   ├── baseline.py              # Baseline 模式使用示例
│   ├── real_cost_example.py     # Real Cost 模式使用示例
│   ├── milp_formulation.md      # MILP 数学建模文档
│   └── results/                 # 结果输出目录（动画、甘特图等）
├── roi/
│   ├── build_roi_table.py       # cost table 生成脚本（需要 tracikpy）
│   ├── plot_roi_coverage.py     # ROI 可视化工具
│   └── results/
│       ├── dual_arm_cost.pkl    # cost table（预计算，IK 真实代价）
│       └── roi_table.pkl        # 完整 ROI 表（含各 yaw 角 IK 解）
├── urdf/
│   └── dual_arm_ik_xy_centered.urdf  # 机械臂 URDF（RealCostPlanner 的唯一机械臂定义来源）
└── dependency/
    └── tracikpy/                # IK 求解器依赖（用于 build_roi_table.py）
```

---

## 环境安装与配置

```bash
# 1. 创建并激活 conda 虚拟环境
conda create -n PickPlan python=3.8
conda activate PickPlan

# 2. 安装基础科学计算与绘图包
pip install "numpy==1.23.5" "scipy==1.10.1" matplotlib
pip install pillow pyparsing shapely alphashape

# 3. 安装机器人与运筹求解相关包
pip install urdfpy
pip install gurobipy
pip install pandas

# 4. 安装本地依赖（tracikpy，用于 build_roi_table.py）
pip install dependency/tracikpy
```

> **注意**：`gurobipy` 需要有效的 Gurobi 许可证（学术版免费申请）。

---

## 快速开始

### 运行 Baseline 模式

```bash
cd planning
python3 baseline.py
# 结果输出到 planning/results/
```

### 运行 Real Cost 模式

```bash
cd planning
python3 real_cost_example.py
# 结果输出到 planning/results/real_cost_example1/
```

### 重新生成 cost table（可选）

如需基于最新 URDF 重新生成 cost table（需要 tracikpy 及 IK 环境）：

```bash
cd roi
python3 build_roi_table.py
# 结果输出到 roi/results/dual_arm_cost.pkl
```

---

## 输出说明

| 文件 | 说明 |
|------|------|
| `*_animation.gif` | 双臂采摘执行过程动画 |
| `*_gantt.png` | 任务调度甘特图 |
| `comparison_summary.txt` | 启发式 vs MILP 对比摘要 |
| `task_distribution.png` | 任务空间分布图 |
| `region_utilization.png` | 区域利用率分析 |
| `performance_comparison.png` | 性能对比图 |
| `task_data.csv` | 原始任务数据 |
| `heuristic_actions.csv` | 启发式调度结果 |
| `milp_actions.csv` | MILP 优化结果 |
| `experiment_config.json` | 实验配置快照 |

---

## 进阶展望

- **强化学习加速求解**：后续将引入 RL 算法对大规模任务场景进行快速求解，并与 MILP 进行对比。
- **三维扩展**：目前固定 z=0.56m，后续可扩展为多高度切片或完整三维规划。
- **在线重规划**：在实际采摘中加入草莓识别、失败重试等在线更新机制。
