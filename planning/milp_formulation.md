# MILP 数学模型：固定停车点下的双臂草莓采摘排程（point-motion / serial-parallel 版）

本文给出当前代码（`planning/realcost_planner.py` 中 `build_milp_model`）真正实现的混合整数线性规划（MILP）模型。

> 注意：这是**当前版本**。早期曾用过 B1–B6 区域 + B2/B5 干涉区的写法，已废弃，本文不再涉及。

---

## 一、直观理解（一句话）

把每个草莓点先离线分成两类：

- **parallel（并行点）**：存在一条**不进危险区**的安全路径，左右臂各摘各的，绝不会撞 → 模型里两臂之间**不加任何先后约束**，可同时进行。
- **serial（串行点）**：只能走**穿过危险区**的路径，两条胳膊不能同时在危险区里 → 模型里把同一危险区（上半 `serial_upper` / 下半 `serial_lower`）的点当作一台**同一时刻只能一个人用的公共机器**，强制互相错开时间。

于是“复杂的在线碰撞检测”被换成了“离线分类 + 排程里的串行约束”，目标是让总完工时间（makespan）最短。

---

## 二、集合与下标

- $\mathcal{F}$：所有待采草莓点（下标 $i, j$）
- $\mathcal{A} = \{L, R\}$：左右臂
- $\mathcal{S}_{\text{up}} \subseteq \mathcal{F}$：标签为 `serial_upper` 的点（上半区危险区内）
- $\mathcal{S}_{\text{lo}} \subseteq \mathcal{F}$：标签为 `serial_lower` 的点（下半区危险区内）
- 其余点（`parallel`）两臂之间无串行约束

> `discard` 点（两臂都够不着）在建模前已被剔除，不进入 $\mathcal{F}$。

---

## 三、参数

- $p_i^a \in \mathbb{R}^+$：臂 $a$ 采摘点 $i$ 并返回篮子的总时间
  $$p_i^a = 2 \cdot c_i^a + b$$
  其中 $c_i^a$ 是 OMPL 求得的**单程**最优运动代价（`optimizer_candidates` 中最小 `best_cost`），乘 2 表示“伸过去 + 收回篮子”，$b$ 是固定处理时间 `base_operation_time`。
- $\delta_i^a \in \{0,1\}$：臂 $a$ 是否可达点 $i$（`allowed_arms`）
- $\mu_i \in \{L, R, \varnothing\}$：点 $i$ 是否被强制指定给某只臂（`must_assign_to`）
- $M$：足够大的常数（代码取 $M = 10000$）

---

## 四、决策变量

- $x_i^a \in \{0,1\}$：点 $i$ 是否分配给臂 $a$
- $t_i \in \mathbb{R}^+$：点 $i$ 的采摘开始时间
- $T \in \mathbb{R}^+$：总完工时间（makespan）
- $o_{ij}^a \in \{0,1\}$：同一臂 $a$ 下，点 $i$ 是否排在 $j$ 之前
- $y_{ij}^g \in \{0,1\}$：在串行组 $g \in \{\text{up}, \text{lo}\}$ 内，点 $i$ 是否排在 $j$ 之前

---

## 五、目标函数

$$\min \; T$$

---

## 六、约束

### 1. 分配约束（每点恰好一只可达的臂）
$$\sum_{a \in \mathcal{A}} \delta_i^a \, x_i^a = 1 \quad \forall i \in \mathcal{F}, \qquad x_i^a = 0 \ \text{ if } \delta_i^a = 0$$

### 2. 强制分配（must_assign_to）
$$x_i^{\mu_i} = 1, \quad x_i^{a} = 0 \ (a \ne \mu_i) \qquad \forall i: \mu_i \in \{L, R\}$$

### 3. 同臂不可重叠（一只手一次只能摘一个）
对同一臂 $a$ 上的任意两点 $i, j$：
$$o_{ij}^a + o_{ji}^a = 1$$
$$t_i + p_i^a \le t_j + M(1 - o_{ij}^a) + M(1 - x_i^a) + M(1 - x_j^a)$$
$$t_j + p_j^a \le t_i + M(1 - o_{ji}^a) + M(1 - x_i^a) + M(1 - x_j^a)$$
（后两个 $M(1-x)$ 项保证：只有当 $i, j$ 都真的分到臂 $a$ 时约束才生效。）

### 4. 串行组约束（危险区 = 共享资源）⭐ 创新点所在
对每个串行组 $g \in \{\text{up}, \text{lo}\}$ 内的任意两点 $i, j$：
$$y_{ij}^g + y_{ji}^g = 1$$
$$t_i + p_i^{a_1} \le t_j + M(1 - y_{ij}^g) + M(1 - x_i^{a_1}) + M(1 - x_j^{a_2}) \quad \forall a_1 \in \text{allowed}(i),\, a_2 \in \text{allowed}(j)$$
$$t_j + p_j^{a_1} \le t_i + M(1 - y_{ji}^g) + M(1 - x_j^{a_1}) + M(1 - x_i^{a_2}) \quad \forall a_1 \in \text{allowed}(j),\, a_2 \in \text{allowed}(i)$$

含义：**无论 $i, j$ 各自分给哪只臂**，同一危险区内的两个串行点都必须一前一后、时间不重叠。
注意 $\mathcal{S}_{\text{up}}$ 与 $\mathcal{S}_{\text{lo}}$ 是两个独立组：上半区串行点之间互斥、下半区串行点之间互斥，但上半区的点和下半区的点**互不约束**（它们在不同的危险子区）。

### 5. 完工时间下界
$$t_i + p_i^a \le T + M(1 - x_i^a) \quad \forall i \in \mathcal{F},\, a \in \text{allowed}(i)$$

---

## 七、关键性质

- **并行点的并行性是“自动获得”的**：对两个 parallel 点（或一个 parallel 点与任意其他点），只要它们分给不同臂，模型里**没有任何**跨臂先后约束，因此可以同时采摘。这正是双臂提速的来源。
- **串行点的安全性是“离线保证”的**：危险区由 `roi/compute_danger_zone.py` 用全关节空间扫掠包络的交集算出，是最坏情况；只要一条路径不进危险区即被判为 safe（→ parallel），因此排程结果天然无碰撞，**无需在线碰撞检测**。

---

## 八、求解与回退策略

代码用一个 **phased 启发式**（`spatial_order_heuristic`）先给出一个可行解作为 MILP 的 warm start，再调用 Gurobi 求最优。

为稳妥起见，`solve_optimization` 里有一条保护：**若 MILP 在时限内找到的解不优于启发式解，则直接保留启发式解**（point-motion 模式下 MILP 理论上不应更差，这条是防止 warm start 未被采纳或时限太短只找到较差 incumbent）。

---

## 九、已知的“简化代价”（写论文/答辩时要诚实说明）

1. 危险区取最坏情况 ⇒ **偏保守**：可能把一些本可并行的点判成串行，损失部分并行度（安全但非全局最优）。
2. 串行组内**一刀切全互斥**：两个串行点即使实际不会撞，也被强制错开。
3. 危险区是 **XY 平面投影**，忽略 z 与时间维度；对“草莓基本在同一竖直面、同一高度”的场景是合理近似。

> 一句话定位贡献：**把碰撞推理从“在线、逐轨迹、实时”搬到“离线、按区域、一次算好”，使在线排程天然无碰撞；代价是结果偏保守。**
