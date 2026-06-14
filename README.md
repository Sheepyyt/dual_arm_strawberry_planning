# 双机械臂草莓采摘规划（固定停车点）

本项目研究**固定停车点下、双机械臂的草莓采摘顺序优化**：在小车停住不动、每摘一个草莓都要回到篮子放下的前提下，决定每个草莓让左手还是右手摘、谁先谁后，使**总完工时间（makespan）最短**。

核心思路（也是创新点）：**用离线算好的“危险区”把碰撞处理掉，避免复杂的在线碰撞检测。**
- 一个草莓点如果存在一条**不进危险区**的安全路径 → 标为 **parallel（并行点）**，两臂可同时摘，绝不会撞。
- 如果只能走**穿过危险区**的路径 → 标为 **serial（串行点）**，同一危险区一次只准一条胳膊进去，必须排队。

数学模型见 [planning/milp_formulation.md](planning/milp_formulation.md)。

---

## 一、项目结构

```text
.
├── config.py                    # ⭐ 所有共享常数（ROI 范围 / 危险区分辨率 / 连杆半径 /
│                                #    上下半区边界 / z / key 格式）的单一事实来源
│
├── roi/                         # 【第 1 段】建表 + 危险区（环境：PickPlan）
│   ├── build_roi_table.py       #   全 ROI 网格 IK 建表（可选的全局参考）
│   ├── compute_danger_zone.py   #   离线计算危险区（两臂扫掠包络的交集）
│   ├── build_points.py          #   对实际采摘点求 IK，分 safe / fallback 解
│   └── results/{grid,danger,points}/
│
├── ompl/                        # 【第 2 段】运动规划 + 点表（环境：ompl）
│   ├── plan.py                  #   运动规划：single（单点调试）/ batch（批量 per-IK）两个子命令
│   ├── point_table.py           #   汇总 + 打标签(parallel/serial_*/discard) → 优化器输入
│   └── results/{single,batch,point_table}/
│
├── planning/                    # 【第 3 段】排程优化（环境：PickPlan，需 Gurobi）
│   ├── realcost_planner.py      #   核心规划器：数据加载 + 启发式 + MILP 求解 + 绘图
│   ├── run_realcost.py          #   ⭐ 主入口
│   ├── milp_formulation.md      #   MILP 数学模型文档
│   └── results/
│
├── urdf/                        # 机械臂模型（流水线只用 dual_arm_ik_xy_centered.urdf）
└── dependency/tracikpy/         # 本地 IK 依赖（TracIK 的 Python 封装）
```

数据流（前一段的产出喂给后一段）：

```
roi/  ──►  ompl/  ──►  planning/
建表+危险区   求路径+打标签    排最优顺序
```

---

## 二、环境构建

本项目建议用**两个 conda 环境**，避免 Python 版本与 OMPL 绑定冲突：

- `PickPlan`：运行 `roi/` 和 `planning/`
- `ompl`：运行 `ompl/`

### 1. PickPlan 环境（Python 3.8）

```bash
conda create -n PickPlan python=3.8
conda activate PickPlan

pip install "numpy==1.23.5" "scipy==1.10.1" matplotlib
pip install pillow pyparsing shapely alphashape
pip install urdfpy pandas gurobipy
pip install dependency/tracikpy     # 本地 IK 依赖
```

- `gurobipy` 需要可用的 **Gurobi 许可证**（只有 `planning/` 求解时才用到）。
- `roi/*.py` 和 `planning/run_realcost.py` 都在这个环境运行。

### 2. ompl 环境（建议 Python 3.10）

```bash
conda create -n ompl python=3.10
conda activate ompl

pip install ompl
pip install numpy scipy matplotlib pandas shapely urdfpy
```

- `ompl/*.py` 在这个环境运行。

### 环境切换规则

```bash
conda activate PickPlan   # 运行 roi/ 和 planning/ 前
conda activate ompl       # 运行 ompl/ 前
```

---

## 三、完整工作流（推荐按顺序跑）

### Step 1. 全 ROI 建表（可选，仅作全局参考图）
```bash
conda activate PickPlan
python roi/build_roi_table.py
```
输出：`roi/results/grid/`（`roi_table.pkl`、`dual_arm_cost.pkl`、覆盖图）

### Step 2. 计算危险区 ⭐
```bash
conda activate PickPlan
python roi/compute_danger_zone.py
```
输出：`roi/results/danger/danger_zone_data.npz` + 包络图

### Step 3. 生成采摘点集 + 分 safe/fallback 解
```bash
conda activate PickPlan
python roi/build_points.py \
  --timeout 0.035 --global-yaw-count 17 \
  --n-global-random-seeds 8 --n-home-perturb-seeds 4 \
  --max-attempts-per-yaw 12 --max-solutions-per-yaw 3 --max-safe-solutions 8
```
输出：`roi/results/points/roi_table_selected_points.pkl` + 可达性图

### Step 4. 单点调试（可选）
```bash
conda activate ompl
python ompl/plan.py single --arm L --key 0.000_0.300_0.560 --mode auto
```
`--mode`：`safe` 只用安全终点 / `free` 只用穿危险区终点 / `auto` 自动二选一。
输出：`ompl/results/single/*.png`

### Step 5. 批量 per-IK 运动规划
```bash
conda activate ompl
python ompl/plan.py batch --mode auto --arms both \
  --solve-time 1.0 --num-trials 5 --planner RRTConnect --save-figures
```
输出：`ompl/results/batch/`（`per_ik.pkl`、`per_ik.csv`、`summary.txt`、`figures/`）

后台跑（耗时较长时）：
```bash
nohup python ompl/plan.py batch --mode auto --arms both --solve-time 1.0 \
  --num-trials 5 --planner RRTConnect --save-figures \
  > ompl/results/batch/nohup_batch.log 2>&1 &
```

### Step 6. 生成优化器输入点表（打标签 ⭐）
```bash
conda activate ompl
python ompl/point_table.py
```
输出：`ompl/results/point_table/point_table.pkl` + csv + summary + 可视化。
**标签规则**：任一臂有 safe → `parallel`；只有 free → `serial_upper`/`serial_lower`（按上下半区）；都没有 → `discard`。

### Step 7. 运行排程优化（启发式 + MILP）⭐
```bash
conda activate PickPlan
python planning/run_realcost.py
```
输出：`planning/results/real_cost_example_all_points/`（甘特图、任务地图、动画、`comparison_summary.txt`）

---

## 四、各模块输入 / 输出速查

| 脚本 | 环境 | 主要输入 | 主要输出 |
|---|---|---|---|
| `roi/build_roi_table.py` | PickPlan | URDF | `roi/results/grid/` |
| `roi/compute_danger_zone.py` | PickPlan | URDF | `roi/results/danger/danger_zone_data.npz` |
| `roi/build_points.py` | PickPlan | URDF + danger npz | `roi/results/points/roi_table_selected_points.pkl` |
| `ompl/plan.py single` | ompl | points pkl + danger npz | `ompl/results/single/*.png` |
| `ompl/plan.py batch` | ompl | points pkl + danger npz | `ompl/results/batch/per_ik.pkl` |
| `ompl/point_table.py` | ompl | `per_ik.pkl` | `ompl/results/point_table/point_table.pkl` |
| `planning/run_realcost.py` | PickPlan | `point_table.pkl` + URDF | `planning/results/...` |

---

## 五、几个关键概念（通俗版）

- **危险区（danger zone）**：让左臂在所有可能姿态下“扫”出它能占据的全部空间，右臂同理，两者**重叠**的那块就是危险区——只有在这里两条胳膊才**可能**撞。由 `compute_danger_zone.py` 一次性离线算好。
- **safe 路径 / free 路径**：伸向某草莓时，全程**不进**危险区的叫 safe；只能**穿过**危险区的叫 free（也叫 fallback）。
- **parallel / serial 标签**：有 safe 路径的点 = 并行点（两臂可同时摘）；只有 free 路径的点 = 串行点（按上/下半区分成 `serial_upper` / `serial_lower`，同一区必须排队）。
- **处理时间** `p = 2 × 单程代价 + 固定处理时间`：乘 2 是因为“伸过去摘 + 收回篮子放下”。

---

## 六、代码组织约定

- **共享常数只改一处**：ROI 扫描范围、危险区分辨率、连杆半径、上下半区边界、采摘高度 z、点 key 格式，全部在 [config.py](config.py)。`roi/`、`ompl/`、`planning/` 都从这里取，不再各自硬编码。
- **single / batch 合成一个文件**：原来 `single.py` 与 `batch.py` 重复了大量 OMPL 公共代码，现已合并为 [ompl/plan.py](ompl/plan.py)，用 `single` / `batch` 两个子命令区分，公共逻辑只写一份。
- **planning 一个模块一个文件**：核心规划器（数据加载 + 启发式 + MILP + 绘图）全在 [planning/realcost_planner.py](planning/realcost_planner.py)，入口在 `run_realcost.py`（原先挤在一个 2000 行大文件里、且夹带已废弃的 Baseline，均已删除）。
