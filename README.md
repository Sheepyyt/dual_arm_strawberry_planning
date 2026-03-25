# 在固定停车点下的双机械臂草莓采摘规划

旨在研究在小车处于固定停车点时，双机械臂采摘草莓的路径与任务规划，以求得到最优的草莓采摘顺序，使总完工时间最短。

## 核心设定

1. **采摘-放置循环（Pick-and-Place）**：
   每采摘完一个草莓，机械臂都需要将草莓放置到小车上的篮子中。因此我们设定，机械臂每次采摘一个草莓后都会返回到同一个初始位置（篮子位置）。处理单个草莓的时间包含了采摘以及放置后的固定时间消耗。
   
2. **干涉区与非干涉区划分（防碰撞简化）**：
   为了兼顾不发生碰撞和提高求解效率（避开计算极其复杂的实时三维碰撞检测），本研究将草莓所在的生长空间极度简化：将其划分为几个正方形网格，并人为设定了“干涉区”和“非干涉区”。
   - **非干涉区**：双臂可互不影响地并行采摘。
   - **干涉区**：严格规定双臂在干涉区内必须保持“串行”作业，绝对禁止双臂同时进入该区域。

3. **时间估算**：
   在现有的代码模型中，使用了空间上的欧式距离来近似代表单程的移动时间。

4. **进阶对比与未来展望**：
   目前使用的是基于 MILP (Mixed Integer Linear Programming) 的精确求解方法与 heuristic 启发式规则方法。后续我们将引入**强化学习 (Reinforcement Learning)** 算法进行加速求解，并将其解算速度与现有 MILP 等方法进行深入对比。

## 目录结构与核心文件

- `planning/milp_formulation.md`: 本研究使用的 MILP 数学建模公式推导与说明。
- `planning/dual_arm_planner.py`: 主规划器逻辑，包含区域定义、时间测算和基于 Gurobi 的 MILP 问题构建等。
- `planning/baseline.py`: 针对此问题的基础调用演示与算法执行脚本。所有结果将生成在 `planning/results` 下。
- `roi/build_roi_table.py`: 工作空间可达性与 ROI (Region of Interest) 表格生成脚本，结果输出极至 `roi/results`。
- `roi/plot_roi_coverage.py`: 对应 ROI 表格的可视化与图形作图工具。

## 环境安装与配置 (Installation)

本项目依赖以下环境，请按顺序配置 Python 环境并安装相关包：

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

# 4. 安装本地依赖 (tracikpy)
pip install dependency/tracikpy
```
