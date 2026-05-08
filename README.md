# 双机械臂草莓采摘规划（固定停车点）

本项目面向**固定停车点**场景下的双机械臂草莓采摘顺序优化。当前工程保留两种模式：

- **Baseline 模式**：原始、简化的区域规则与欧氏代价模型
- **Real Cost 模式（point-motion）**：基于 ROI + OMPL + 点级轨迹安全规则的真实代价模式

---

## 当前工程结构

```text
.
├── planning/
│   ├── baseline.py
│   ├── dual_arm_planner.py
│   ├── milp_formulation.md
│   ├── real_cost_example.py
│   └── results/
├── roi/
│   ├── build_roi_table.py        # 全 ROI 网格建表（并顺便画 dual_arm_roi_coverage.png）
│   ├── build_points.py           # 手工/中密度/调试点集建表（并顺便画 selected_points_reachability.png）
│   ├── compute_danger_zone.py    # 危险区计算（并顺便画 danger_zone_envelopes.png）
│   └── results/
│       ├── grid/
│       ├── danger/
│       └── points/
├── ompl/
│   ├── single.py                 # 单点单臂调试
│   ├── batch.py                  # 批量 per-IK OMPL 规划
│   ├── point_table.py            # per-IK 结果整理为优化器输入，并自动画图
│   └── results/
│       ├── single/
│       ├── batch/
│       └── point_table/
└── urdf/
```

---

## 各模块职责

### 1. ROI 模块

#### `roi/build_roi_table.py`
用于**全 ROI 网格**建表：
- 输出：`roi/results/grid/roi_table.pkl`
- 输出：`roi/results/grid/dual_arm_cost.pkl`
- 输出：`roi/results/grid/dual_arm_roi_coverage.png`
- 若已存在结果且 `force_regenerate=False`，则直接读取数据并重画图

#### `roi/build_points.py`
用于**手工点集 / 中密度采样 / 调试点集**建表：
- 输出：`roi/results/points/roi_table_selected_points.pkl`
- 输出：`roi/results/points/selected_points_reachability.png`
- 需要危险区数据：`roi/results/danger/danger_zone_data.npz`
- 若已存在结果，默认直接读取并重画图

#### `roi/compute_danger_zone.py`
用于离线计算危险区：
- 输出：`roi/results/danger/danger_zone_data.npz`
- 输出：`roi/results/danger/danger_zone_envelopes.png`
- 若已有数据，则直接读取并重画图

---

### 2. OMPL 模块

#### `ompl/single.py`
单点、单臂调试脚本。适合检查：
- 某个点、某只臂的 safe / free 规划是否成功
- 候选 goal 姿态与最终 best path 的关系

默认输入：
- `roi/results/points/roi_table_selected_points.pkl`
- `roi/results/danger/danger_zone_data.npz`

输出到：
- `ompl/results/single/`

#### `ompl/batch.py`
批量对所有点、所有臂、所有 IK 解执行 OMPL：
- 按 `(point, arm, ik_index)` 粒度逐个规划
- 对同一 `(key, arm)`，只保存**成功且代价最小**的那一个 IK 的图

默认输入：
- `roi/results/points/roi_table_selected_points.pkl`
- `roi/results/danger/danger_zone_data.npz`

输出到：
- `ompl/results/batch/per_ik.pkl`
- `ompl/results/batch/per_ik.csv`
- `ompl/results/batch/summary.txt`
- `ompl/results/batch/figures/`

#### `ompl/point_table.py`
把 per-IK 结果整理成**优化器直接读取的 point-level 表**，并自动可视化：
- 输出：`ompl/results/point_table/point_table.pkl`
- 输出：`ompl/results/point_table/point_table.csv`
- 输出：`ompl/results/point_table/summary.txt`
- 输出：`ompl/results/point_table/visualizations/*.png`
- 若 point table 已存在，则默认直接读取并重画图

---

## Real Cost（point-motion）模式规则

当前 Real Cost 模式已经不再使用旧的 B1-B6 正方形区域逻辑，而改为使用 point-level motion table。

### 点级标签规则

#### `parallel`
只要某点至少存在一条 **safe** 结果（任一臂），该点就是 `parallel`：
- 若 L/R 都有 safe：`allowed_arms=[L,R]`
- 若只有 L 有 safe：`allowed_arms=[L]`, `must_assign_to=L`
- 若只有 R 有 safe：`allowed_arms=[R]`, `must_assign_to=R`

并且：
- 优化器**只允许使用 safe 候选**
- free 候选全部丢弃，不参与优化

#### `serial_upper` / `serial_lower`
若两臂都没有 safe，但存在 free：
- 上半区：`serial_upper`
- 下半区：`serial_lower`

并且：
- 优化器**只允许使用 free 候选**
- 若仅一只臂有 free，则 `must_assign_to` 为该臂

#### `discard`
若两臂既无 safe 也无 free，则丢弃。

---

## Real Cost 模式中的调度约束

- 同一只机械臂上的任务不能重叠
- 两个 `serial_upper` 任务不能同时执行
- 两个 `serial_lower` 任务不能同时执行
- `parallel` 与 `serial_upper/serial_lower` 可以并行，只要满足分配臂约束
- 不再使用 B1-B6 作为可视化与约束来源

---

## 启发式 warm start（phased heuristic）

当前 point-motion Real Cost 模式使用的启发式规则：

### 左臂顺序
1. `serial_upper`
2. 上半 `parallel`
3. 下半 `parallel`
4. `serial_lower`

### 右臂顺序
1. `serial_lower`
2. 下半 `parallel`
3. 上半 `parallel`
4. `serial_upper`

左右臂从 `t=0` 同时开始工作。

---

## 推荐工作流

### A. 全 ROI 网格建表
```bash
python roi/build_roi_table.py
```

### B. 计算危险区
```bash
python roi/compute_danger_zone.py
```

### C. 生成手工/调试点集 ROI 表
```bash
python roi/build_points.py   --timeout 0.035   --global-yaw-count 17   --n-global-random-seeds 8   --n-home-perturb-seeds 4   --max-attempts-per-yaw 12   --max-solutions-per-yaw 3   --max-safe-solutions 8
```

### D. 单点调试
```bash
python ompl/single.py --arm L --key 0.000_0.300_0.560 --mode auto
```

### E. 批量 per-IK OMPL
```bash
python ompl/batch.py --mode auto --arms both --solve-time 1.0 --num-trials 5 --planner RRTConnect --save-figures
```

### F. 生成优化器输入 point table
```bash
python ompl/point_table.py
```

### G. 运行 point-motion Real Cost 模式
```bash
python planning/real_cost_example.py
```

---

## 结果目录说明

### `roi/results/grid/`
全 ROI 网格建表结果

### `roi/results/danger/`
危险区数据与图

### `roi/results/points/`
手工点集 / 中密度点集建表结果

### `ompl/results/single/`
单点调试结果

### `ompl/results/batch/`
批量 per-IK 规划结果

### `ompl/results/point_table/`
点级优化器输入与可视化结果

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
