
# MILP Formulation for Dual-Arm Strawberry Harvesting Robot

This document presents a complete Mixed-Integer Linear Programming (MILP) formulation for scheduling a dual-arm harvesting robot in a fixed location. The formulation supports parallel harvesting in non-interference zones and enforces serialized picking in interference zones where arm-robot collisions may occur.

---

## Problem Description

Given a set of strawberries, each with a known harvesting time and location, and two robot arms (Left `L` and Right `R`), assign each fruit to one arm and schedule the harvesting operations to **minimize the total harvesting time**. The following conditions apply:

- Each strawberry must be harvested by exactly one arm.
- Arms have different reachability zones.
- In **non-interference zones**, arms can harvest simultaneously.
- In **interference zones**, if both arms attempt to harvest at the same time, collisions occur and must be avoided.
- The robot base is fixed; no repositioning is allowed.

---

## Sets and Indices

- $\mathcal{F}$ : Set of all strawberries (indexed by $i$)
- $\mathcal{A} = \{L, R\}$: Set of arms (Left, Right)
- $\mathcal{I} \subset \mathcal{F}$: Strawberries in the **interference zone**
- $\mathcal{N} = \mathcal{F} \setminus \mathcal{I}$: Strawberries in **non-interference zones**

---

## Parameters

- $p_i^a \in \mathbb{R}^+$: Time for arm $a$ to harvest fruit $i$ and return
- $\delta_i^a \in \{0, 1\}$: 1 if arm $a$ can reach fruit $i$
- $M$: A large constant ($M \gg \max p_i^a$)

---

## Decision Variables

- $x_i^a \in \{0,1\}$: 1 if fruit $i$ is assigned to arm $a$
- $t_i \in \mathbb{R}^+$: Start time of harvesting fruit $i$
- $T \in \mathbb{R}^+$: Makespan (total harvesting time)
- $o_{i,j}^a \in \{0,1\}$: 1 if fruit $i$ is picked before $j$ by arm $a$
- $y_{i,j} \in \{0,1\}$: 1 if fruit $i \in \mathcal{I}$ is picked before $j \in \mathcal{I}$
- $z_{i,j} \in \{0,1\}$: 1 if fruit $i$ is picked by $L$ and $j$ by $R$

---

## Objective

$$
\min T
$$

---

## Constraints

### 1. Assignment Constraint

Each fruit is picked by exactly one arm:
$$
\sum_{a \in \mathcal{A}} x_i^a = 1 \quad \forall i \in \mathcal{F}
$$

### 2. Reachability Constraint

Fruits can only be picked by reachable arms:
$$
x_i^a \leq \delta_i^a \quad \forall i \in \mathcal{F}, a \in \mathcal{A}
$$

### 3. Arm-Specific Sequencing

For fruits assigned to the same arm:
$$
t_i + p_i^a \leq t_j + M(1 - o_{i,j}^a) \\
t_j + p_j^a \leq t_i + M o_{i,j}^a \\
o_{i,j}^a + o_{j,i}^a = 1
$$

### 4. Interference Constraints (Conditional Disjunctive)

Only enforced when both arms pick in $\mathcal{I}$:

Binary linkage:
$$
z_{i,j} \leq x_i^L, \quad z_{i,j} \leq x_j^R, \quad z_{i,j} \geq x_i^L + x_j^R - 1
$$

Disjunctive ordering:
$$
y_{i,j} + y_{j,i} = 1 \\
t_i + p_i^L \leq t_j + M(1 - z_{i,j} y_{i,j}) \\
t_j + p_j^R \leq t_i + M(1 - z_{i,j} y_{j,i})
$$

### 5. Makespan Constraint

Ensure all harvesting ends by $T$:
$$
t_i + p_i^a \leq T + M (1 - x_i^a) \quad \forall i \in \mathcal{F}, a \in \mathcal{A}
$$

---

## Key Insight

- **Parallel harvesting** is automatically allowed for fruits assigned to different arms **in non-interference zones**, since **no sequencing constraint is enforced** between them.
- This omission is deliberate and enables full utilization of both arms when possible.

---

## Conclusion

This MILP formulation supports exact scheduling of dual-arm robots for strawberry harvesting, capturing collision-avoidance in interference regions and maximizing parallelism elsewhere.