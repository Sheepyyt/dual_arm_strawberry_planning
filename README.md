# 双机械臂草莓采摘规划（固定停车点）

本项目研究固定停车点下的双机械臂草莓采摘顺序优化。当前工程保留两种模式：

- **Baseline 模式**：快速、简化的对照模式。它使用 B1–B6 区域规则和欧氏距离代价，不考虑轨迹级安全，只适合做基础对比和快速演示。
- **Real Cost / point-motion 模式**：当前主模式。它先做 ROI 建表、危险区计算、OMPL 路径规划，再把结果整理成 point table，最后交给优化器调度。这个模式考虑了“点级 safe / free 路径”与上半/下半串行点集规则。

---

## 一、项目结构

```text
.
├── planning/
│   ├── baseline.py
│   ├── dual_arm_planner.py
│   ├── milp_formulation.md
│   ├── real_cost_example.py
│   └── results/
├── roi/
│   ├── build_roi_table.py
│   ├── build_points.py
│   ├── compute_danger_zone.py
│   └── results/
│       ├── grid/
│       ├── danger/
│       └── points/
├── ompl/
│   ├── single.py
│   ├── batch.py
│   ├── point_table.py
│   └── results/
│       ├── single/
│       ├── batch/
│       └── point_table/
├── urdf/
└── dependency/
```

---

## 二、环境构建

本项目建议使用两个 conda 环境：

- `PickPlan`：运行 `roi/` 和 `planning/` 模块
- `ompl`：运行 `ompl/` 模块

这样可以避免 Python 3.8 主环境和 OMPL Python bindings 的版本冲突。

### 1. 构建 PickPlan 环境（Python 3.8）

```bash
conda create -n PickPlan python=3.8
conda activate PickPlan

# 基础科学计算与绘图
pip install "numpy==1.23.5" "scipy==1.10.1" matplotlib
pip install pillow pyparsing shapely alphashape

# 机器人与优化相关
pip install urdfpy
pip install gurobipy
pip install pandas

# 本地 IK 依赖
pip install dependency/tracikpy
```

说明：
- `gurobipy` 需要可用的 Gurobi 许可证。
- `roi/build_roi_table.py`、`roi/build_points.py`、`planning/real_cost_example.py` 都在这个环境下运行。

### 2. 构建 ompl 环境（建议 Python 3.10）

```bash
conda create -n ompl python=3.10
conda activate ompl

pip install ompl
pip install numpy scipy matplotlib pandas shapely urdfpy
```

说明：
- `ompl/single.py`、`ompl/batch.py`、`ompl/point_table.py` 在这个环境下运行。
- 如果你的 OMPL 环境名不是 `ompl`，把下面命令里的环境名替换成你自己的即可。

### 3. 环境切换规则

- 运行 `roi/` 和 `planning/` 脚本前：
  ```bash
  conda activate PickPlan
  ```
- 运行 `ompl/` 脚本前：
  ```bash
  conda activate ompl
  ```

---

## 三、各脚本作用、输入和输出

下面按模块说明。你只需要关心“输入是什么、输出到哪里、在哪个环境运行”。

---

## 四、planning 模块

### 1. `planning/baseline.py`

**作用**：
- 运行 Baseline 模式示例。
- Baseline 只使用简化区域规则和欧氏距离代价。

**输入**：
- 不依赖外部结果文件。
- 脚本内部自己随机生成任务，或使用脚本里写死的示例点。
- 机械臂区域规则来自 `planning/dual_arm_planner.py` 中的默认 B1–B6 配置。

**输出**：
- `planning/results/example1_results/`
- `planning/results/example2_results/`

**运行环境**：`PickPlan`

**运行方式**：
```bash
python planning/baseline.py
```

---

### 2. `planning/real_cost_example.py`

**作用**：
- 运行 point-motion 版 Real Cost 模式示例。
- 读取 point table，并调用优化器。

**输入**：
- 默认读取：
  - `ompl/results/point_table/point_table.pkl`
- 同时会读取：
  - `urdf/dual_arm_ik_xy_centered.urdf`

**输出**：
- `planning/results/real_cost_example_all_points/`
- `planning/results/real_cost_example_sampled/`
- `planning/results/real_cost_example_specified/`

**运行环境**：`PickPlan`

**运行方式**：
```bash
python planning/real_cost_example.py
```

---

### 3. `planning/dual_arm_planner.py`

**作用**：
- 包含 BaselinePlanner 和 point-motion 版 RealCostPlanner。
- 这是核心优化器文件，通常被 `baseline.py` 和 `real_cost_example.py` 调用。

**输入**：
- BaselinePlanner：不依赖外部文件。
- RealCostPlanner 默认读取：
  - `ompl/results/point_table/point_table.pkl`
  - `urdf/dual_arm_ik_xy_centered.urdf`

**输出**：
- 自身不直接作为入口脚本使用；输出由调用它的脚本保存。

**运行环境**：`PickPlan`

---

## 五、roi 模块

### 1. `roi/build_roi_table.py`

**作用**：
- 对整块 ROI 网格做 IK 建表。
- 生成全 ROI 的可达性表和 cost table。
- 并绘制 `dual_arm_roi_coverage.png`。

**输入**：
- `urdf/dual_arm_ik_xy_centered.urdf`
- `dependency/tracikpy`
- 脚本内部 `CONFIG` 中的扫描范围、步长、yaw 搜索参数

**输出**：
- `roi/results/grid/roi_table.pkl`
- `roi/results/grid/dual_arm_cost.pkl`
- `roi/results/grid/dual_arm_roi_coverage.png`

**特别说明**：
- 如果 `roi_table.pkl` 和 `dual_arm_cost.pkl` 已存在，默认不会重新做 IK 求解。
- 但现在即使只缺 `dual_arm_roi_coverage.png`，再次运行本脚本也会直接补画该图。

**运行环境**：`PickPlan`

**运行方式**：
```bash
python roi/build_roi_table.py
```

---

### 2. `roi/compute_danger_zone.py`

**作用**：
- 离线计算危险区 envelope。
- 输出危险区数据和图。

**输入**：
- `urdf/dual_arm_ik_xy_centered.urdf`
- 脚本内部采样参数（如 `N_SAMPLES`、`GRID_RES` 等）

**输出**：
- `roi/results/danger/danger_zone_data.npz`
- `roi/results/danger/danger_zone_envelopes.png`

**运行环境**：`PickPlan`

**运行方式**：
```bash
python roi/compute_danger_zone.py
```

---

### 3. `roi/build_points.py`

**作用**：
- 对手工指定点集、中等密度采样点集或调试点集做建表。
- 当前是你后续中等密度采样和逐步加密最常用的入口。
- 会同时做可达性统计和可视化。

**输入**：
- `urdf/dual_arm_ik_xy_centered.urdf`
- `roi/results/danger/danger_zone_data.npz`
- 脚本参数中的 `x-values`、`y-values`、`z`、yaw 搜索参数、seed 搜索参数

**输出**：
- `roi/results/points/roi_table_selected_points.pkl`
- `roi/results/points/selected_points_reachability.png`

**运行环境**：`PickPlan`

**运行方式（当前常用的一组轻量广覆盖参数）**：
```bash
python roi/build_points.py \
  --timeout 0.035 \
  --global-yaw-count 17 \
  --n-global-random-seeds 8 \
  --n-home-perturb-seeds 4 \
  --max-attempts-per-yaw 12 \
  --max-solutions-per-yaw 3 \
  --max-safe-solutions 8
```

---

## 六、ompl 模块

### 1. `ompl/single.py`

**作用**：
- 单点、单臂调试 OMPL。
- 用来检查某个点某只臂的 safe / free 路径、候选终点姿态和最终 best path。

**输入**：
- 默认读取：
  - `roi/results/points/roi_table_selected_points.pkl`
  - `roi/results/danger/danger_zone_data.npz`
- 同时读取：
  - `urdf/dual_arm_ik_xy_centered.urdf`

**输出**：
- `ompl/results/single/*.png`

**运行环境**：`ompl`

**运行方式**：
```bash
python ompl/single.py --arm L --key 0.000_0.300_0.560 --mode auto
```

模式说明：
- `--mode safe`：只用 safe 终点解
- `--mode free`：只用 free/fallback 终点解
- `--mode auto`：点不在危险区时优先 safe，否则用 free

---

### 2. `ompl/batch.py`

**作用**：
- 对所有点、所有臂、所有 IK 解逐个执行 OMPL。
- 粒度是 `(point, arm, ik_index)`。
- 对同一个 `(key, arm)`，只保存“成功且 best_cost 最小”的那一个 IK 的图。

**输入**：
- 默认读取：
  - `roi/results/points/roi_table_selected_points.pkl`
  - `roi/results/danger/danger_zone_data.npz`
- 同时读取：
  - `urdf/dual_arm_ik_xy_centered.urdf`

**输出**：
- `ompl/results/batch/per_ik.pkl`
- `ompl/results/batch/per_ik.csv`
- `ompl/results/batch/summary.txt`
- `ompl/results/batch/figures/*.png`
- `ompl/results/batch/nohup_*.log`（如果你用 nohup）

**运行环境**：`ompl`

**前台运行示例**：
```bash
python ompl/batch.py \
  --mode auto \
  --arms both \
  --solve-time 1.0 \
  --num-trials 5 \
  --planner RRTConnect \
  --save-figures
```

**后台运行示例（nohup）**：
```bash
nohup python ompl/batch.py \
  --mode auto \
  --arms both \
  --solve-time 1.0 \
  --num-trials 5 \
  --planner RRTConnect \
  --save-figures \
  > ompl/results/batch/nohup_batch.log 2>&1 &
```

---

### 3. `ompl/point_table.py`

**作用**：
- 把 `batch.py` 的 per-IK 结果整理为优化器直接读取的 point-level table。
- 同时自动画图，方便查看最终传入优化器的数据。

**输入**：
- 默认读取：
  - `ompl/results/batch/per_ik.pkl`

**输出**：
- `ompl/results/point_table/point_table.pkl`
- `ompl/results/point_table/point_table.csv`
- `ompl/results/point_table/summary.txt`
- `ompl/results/point_table/visualizations/*.png`

**运行环境**：`ompl`

**运行方式**：
```bash
python ompl/point_table.py
```

---

## 七、推荐工作流

如果你要完整跑一遍 point-motion Real Cost 模式，推荐顺序如下：

### Step 1. 全 ROI 建表（可选）
```bash
conda activate PickPlan
python roi/build_roi_table.py
```

### Step 2. 计算危险区
```bash
conda activate PickPlan
python roi/compute_danger_zone.py
```

### Step 3. 生成中等密度 / 调试点集
```bash
conda activate PickPlan
python roi/build_points.py \
  --timeout 0.035 \
  --global-yaw-count 17 \
  --n-global-random-seeds 8 \
  --n-home-perturb-seeds 4 \
  --max-attempts-per-yaw 12 \
  --max-solutions-per-yaw 3 \
  --max-safe-solutions 8
```

### Step 4. 单点调试（可选）
```bash
conda activate ompl
python ompl/single.py --arm L --key 0.000_0.300_0.560 --mode auto
```

### Step 5. 批量 per-IK OMPL
```bash
conda activate ompl
python ompl/batch.py --mode auto --arms both --solve-time 1.0 --num-trials 5 --planner RRTConnect --save-figures
```

### Step 6. 生成优化器输入 point table
```bash
conda activate ompl
python ompl/point_table.py
```

### Step 7. 运行 point-motion Real Cost 调度优化
```bash
conda activate PickPlan
python planning/real_cost_example.py
```

---

## 八、Baseline 与 Real Cost 的区别（简单理解）

### Baseline
- 不需要 ROI 建表、OMPL、危险区数据
- 直接随机生成点
- 用欧氏距离近似代价
- 规则简单，适合快速对比

### Real Cost / point-motion
- 先做 IK 建表和危险区分析
- 再用 OMPL 求 safe / free 路径
- 再整理成 point table 给优化器
- 更接近你当前项目最终想要的规则和真实代价

如果你只是想快速跑通流程，用 Baseline。  
如果你想使用当前项目真正的轨迹安全规则，用 Real Cost / point-motion。

---

## 九、结果目录怎么理解

### `roi/results/grid/`
全 ROI 网格建表结果：
- `roi_table.pkl`
- `dual_arm_cost.pkl`
- `dual_arm_roi_coverage.png`

### `roi/results/danger/`
危险区数据与图：
- `danger_zone_data.npz`
- `danger_zone_envelopes.png`

### `roi/results/points/`
手工点 / 中密度点 / 调试点建表结果：
- `roi_table_selected_points.pkl`
- `selected_points_reachability.png`

### `ompl/results/single/`
单点调试结果图

### `ompl/results/batch/`
per-IK 批量路径规划结果

### `ompl/results/point_table/`
最终传入优化器的 point-level 表及其可视化

### `planning/results/`
优化器输出的任务分配图、甘特图、summary

---