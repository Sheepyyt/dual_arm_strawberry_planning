#!/usr/bin/env python3
"""
milp_vs_heuristic_scan.py — "启发式 vs MILP 最优" 随 N 的基线扫描（给 RL 当对照）

对每个规模 N，随机抽 N 个非 discard 采摘点，构成一个实例，比较：
  - 启发式 (spatial_order_heuristic) 的 makespan
  - MILP 真·最优 makespan（load-based 模型，求到证明最优 / MIPGap=0）
量化"启发式相对最优还差多少"。这条差距曲线就是 RL 要去填的空间：
RL 的目标是在 MILP 跑不动的规模上，逼近这里 MILP 给出的最优。

依赖：load-based build_milp_model（数值稳健、中小规模秒证最优）。
makespan 取 model.ObjVal（其 makespan 下界为负载约束，无 big-M，默认容差即精确）。

运行：
    conda activate PickPlan
    python planning/milp_vs_heuristic_scan.py
    # 或自定义： python planning/milp_vs_heuristic_scan.py --N 5,10,20,40 --seeds 20 --time-limit 120
"""
import os
import sys
import csv
import time
import random
import argparse
import statistics

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from realcost_planner import RealCostPlanner
from gurobipy import GRB


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=str, default="5,10,15,20,30,40,50,70",
                   help="逗号分隔的实例规模列表")
    p.add_argument("--seeds", type=int, default=15, help="每个 N 的随机实例个数")
    p.add_argument("--time-limit", type=int, default=120, help="单个 MILP 求解时限(s)")
    p.add_argument("--out-dir", type=str,
                   default=os.path.join(SCRIPT_DIR, "results", "milp_vs_heuristic_scan"))
    return p.parse_args()


def main():
    args = parse_args()
    N_list = [int(x) for x in args.N.split(",") if x.strip()]
    os.makedirs(args.out_dir, exist_ok=True)

    planner = RealCostPlanner()
    all_keys = [pt["key"] for pt in planner.point_payload["points"]
                if not bool(pt.get("discard", False))]
    print(f"[info] available non-discard points: {len(all_keys)}")
    print(f"[info] N={N_list} seeds={args.seeds} time_limit={args.time_limit}s\n")

    rows = []
    per_N = {}
    for N in N_list:
        if N > len(all_keys):
            print(f"[skip] N={N} > available {len(all_keys)}")
            continue
        improvs, times, n_proven = [], [], 0
        for seed in range(args.seeds):
            rng = random.Random(1000 * N + seed)
            keys = rng.sample(all_keys, N)
            planner.load_task_locations([{"key": k} for k in keys])
            heur = planner.spatial_order_heuristic()
            heur_mk = max(a["end"] for a in heur) if heur else 0.0

            m = planner.build_milp_model(heur)
            m.setParam("OutputFlag", 0)
            m.setParam("TimeLimit", args.time_limit)
            m.setParam("MIPGap", 0.0)
            t0 = time.time()
            m.optimize()
            dt = time.time() - t0

            milp_mk = float(m.ObjVal) if m.SolCount > 0 else float("nan")
            proven = (m.status == GRB.OPTIMAL)
            gap = float(m.MIPGap) if m.SolCount > 0 else float("nan")
            improv = (heur_mk - milp_mk) / heur_mk * 100.0 if heur_mk > 0 else 0.0
            if -1e-6 < improv < 0:
                improv = 0.0
            n_proven += int(proven)
            improvs.append(improv)
            times.append(dt)
            rows.append({
                "N": N, "seed": seed, "heuristic_makespan": round(heur_mk, 4),
                "milp_makespan": round(milp_mk, 4), "improvement_pct": round(improv, 4),
                "proven_optimal": proven, "mip_gap_pct": round(gap * 100, 4),
                "solve_time_s": round(dt, 3),
            })
            print(f"  N={N:3d} seed={seed:2d} | heur={heur_mk:7.2f}  milp={milp_mk:7.2f}  "
                  f"improv={improv:5.2f}%  proven={proven}  {dt:5.2f}s")
        per_N[N] = {
            "mean": statistics.mean(improvs), "median": statistics.median(improvs),
            "max": max(improvs), "min": min(improvs),
            "std": statistics.pstdev(improvs) if len(improvs) > 1 else 0.0,
            "n_proven": n_proven, "n": len(improvs),
            "mean_time": statistics.mean(times), "max_time": max(times),
        }
        print()

    # ---- CSV ----
    csv_path = os.path.join(args.out_dir, "scan_per_instance.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # ---- summary ----
    sum_path = os.path.join(args.out_dir, "summary.txt")
    with open(sum_path, "w", encoding="utf-8") as f:
        f.write("MILP(load-based, optimal) vs Heuristic — makespan improvement by instance size\n")
        f.write(f"seeds per N: {args.seeds} | time_limit: {args.time_limit}s\n\n")
        f.write(f"{'N':>4} | {'#proven':>8} | {'mean%':>7} | {'median%':>8} | {'max%':>7} | "
                f"{'std%':>6} | {'mean_t(s)':>9} | {'max_t(s)':>8}\n")
        f.write("-" * 78 + "\n")
        for N in sorted(per_N):
            s = per_N[N]
            f.write(f"{N:>4} | {s['n_proven']:>3}/{s['n']:<4} | {s['mean']:>7.2f} | "
                    f"{s['median']:>8.2f} | {s['max']:>7.2f} | {s['std']:>6.2f} | "
                    f"{s['mean_time']:>9.3f} | {s['max_time']:>8.3f}\n")
    print(open(sum_path, encoding="utf-8").read())

    # ---- figure ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        Ns = sorted(per_N)
        means = [per_N[N]["mean"] for N in Ns]
        mins = [per_N[N]["min"] for N in Ns]
        maxs = [per_N[N]["max"] for N in Ns]
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.fill_between(Ns, mins, maxs, alpha=0.18, color="#5B71B5", label="min–max range")
        ax.plot(Ns, means, "-o", color="#2850a0", lw=2, label="mean improvement")
        for N in Ns:
            ax.annotate(f"{per_N[N]['n_proven']}/{per_N[N]['n']} opt",
                        (N, per_N[N]["max"]), fontsize=7, ha="center", va="bottom", color="gray")
        ax.set_xlabel("instance size N (random strawberries per stop)")
        ax.set_ylabel("makespan improvement of MILP-optimal over heuristic (%)")
        ax.set_title("How far the heuristic is from optimal vs problem size\n(the gap RL aims to close)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")
        fig.tight_layout()
        fig_path = os.path.join(args.out_dir, "improvement_vs_N.png")
        fig.savefig(fig_path, dpi=200, bbox_inches="tight")
        print(f"Saved figure: {fig_path}")
    except Exception as e:
        print(f"[warn] figure skipped: {e}")

    print(f"\nSaved: {csv_path}\nSaved: {sum_path}")


if __name__ == "__main__":
    main()
