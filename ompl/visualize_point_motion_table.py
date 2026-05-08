
#!/usr/bin/env python3
"""
visualize_point_motion_table.py

将 point_motion_table_selected_points_summary.csv / .pkl 中最终传入优化器的数据可视化。
建议放在项目根目录的 ompl/ 下运行。

默认输入:
    ompl/results/point_motion_table/point_motion_table_selected_points_summary.csv

默认输出目录:
    ompl/results/point_motion_table/visualizations/
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input-csv",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "results", "point_motion_table", "point_motion_table_selected_points_summary.csv"
        ),
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "results", "point_motion_table", "visualizations"
        ),
    )
    p.add_argument("--annotate-counts", action="store_true", help="在点旁标注 safe/free 数量")
    p.add_argument("--annotate-costs", action="store_true", help="在点旁标注优化器使用的单程代价")
    p.add_argument("--dpi", type=int, default=220)
    return p


def _save(fig, path, dpi):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def _base_axes(title):
    fig, ax = plt.subplots(figsize=(8.2, 8.0))
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.48, 0.48)
    ax.set_ylim(-0.58, 0.58)
    return fig, ax


def plot_label_map(df, out_path, dpi):
    fig, ax = _base_axes("Point labels for optimizer")
    # 不同 label 用不同 marker；颜色使用 matplotlib 默认循环
    spec = [
        ("parallel", "o"),
        ("serial_upper", "s"),
        ("serial_lower", "^"),
        ("discard", "x"),
    ]
    for label, marker in spec:
        sub = df[df["label"] == label]
        if len(sub) > 0:
            ax.scatter(sub["x"], sub["y"], marker=marker, s=90, label=label)

    # 标出 must_assign_to
    for _, r in df.iterrows():
        txt = ""
        if pd.notna(r["must_assign_to"]) and str(r["must_assign_to"]).strip() not in ("", "None", "nan"):
            txt = f'{r["must_assign_to"]}'
        if txt:
            ax.text(r["x"] + 0.012, r["y"] + 0.012, txt, fontsize=8)

    ax.legend()
    _save(fig, out_path, dpi)


def plot_allowed_arms_map(df, out_path, dpi):
    fig, ax = _base_axes("Allowed arms / assignment")
    groups = [
        ("L", "o"),
        ("R", "s"),
        ("L,R", "^"),
    ]
    # 标准化 allowed_arms 字符串
    norm = df.copy()
    norm["allowed_arms_norm"] = norm["allowed_arms"].astype(str).str.replace(" ", "", regex=False)
    for tag, marker in groups:
        sub = norm[norm["allowed_arms_norm"] == tag]
        if len(sub) > 0:
            ax.scatter(sub["x"], sub["y"], marker=marker, s=90, label=f"allowed={tag}")

    for _, r in norm.iterrows():
        label = str(r["label"])
        if label.startswith("serial"):
            ax.text(r["x"] + 0.010, r["y"] - 0.018, "S", fontsize=8)
        elif label == "parallel":
            ax.text(r["x"] + 0.010, r["y"] - 0.018, "P", fontsize=8)

    ax.legend()
    _save(fig, out_path, dpi)


def plot_safe_free_counts(df, out_path, dpi):
    fig, ax = _base_axes("Safe / free counts at each point")
    # 先按 label 打底
    for label, marker in [("parallel", "o"), ("serial_upper", "s"), ("serial_lower", "^"), ("discard", "x")]:
        sub = df[df["label"] == label]
        if len(sub) > 0:
            ax.scatter(sub["x"], sub["y"], marker=marker, s=60, label=label)

    # 左右臂分别标注
    for _, r in df.iterrows():
        x, y = r["x"], r["y"]
        txt_L = f'L {int(r["safe_count_L"])}/{int(r["free_count_L"])}'
        txt_R = f'R {int(r["safe_count_R"])}/{int(r["free_count_R"])}'
        ax.text(x + 0.012, y + 0.012, txt_L, fontsize=6)
        ax.text(x + 0.012, y - 0.020, txt_R, fontsize=6)

    ax.legend()
    _save(fig, out_path, dpi)


def plot_optimizer_cost_map(df, out_path, dpi):
    fig, ax = _base_axes("Optimizer single-trip cost (L / R)")
    # 打底
    ax.scatter(df["x"], df["y"], s=50, label="points")

    for _, r in df.iterrows():
        x, y = r["x"], r["y"]
        txts = []
        if pd.notna(r["optimizer_best_cost_L"]):
            txts.append(f'L {r["optimizer_best_cost_L"]:.2f}')
        if pd.notna(r["optimizer_best_cost_R"]):
            txts.append(f'R {r["optimizer_best_cost_R"]:.2f}')
        if txts:
            ax.text(x + 0.012, y + 0.012, "\n".join(txts), fontsize=6)

    ax.legend()
    _save(fig, out_path, dpi)


def plot_compact_overview(df, out_path, dpi, annotate_counts=False, annotate_costs=False):
    fig, ax = _base_axes("Compact overview for optimizer input")

    # 分类底图
    spec = [
        ("parallel", "o"),
        ("serial_upper", "s"),
        ("serial_lower", "^"),
        ("discard", "x"),
    ]
    for label, marker in spec:
        sub = df[df["label"] == label]
        if len(sub) > 0:
            ax.scatter(sub["x"], sub["y"], marker=marker, s=85, label=label)

    # 简洁文本：must_assign_to + 可选统计
    for _, r in df.iterrows():
        x, y = r["x"], r["y"]
        parts = []
        ma = str(r["must_assign_to"])
        if ma not in ("", "None", "nan"):
            parts.append(f"M={ma}")
        if annotate_counts:
            parts.append(f'L {int(r["safe_count_L"])}/{int(r["free_count_L"])}')
            parts.append(f'R {int(r["safe_count_R"])}/{int(r["free_count_R"])}')
        if annotate_costs:
            cost_parts = []
            if pd.notna(r["optimizer_best_cost_L"]):
                cost_parts.append(f'L {r["optimizer_best_cost_L"]:.2f}')
            if pd.notna(r["optimizer_best_cost_R"]):
                cost_parts.append(f'R {r["optimizer_best_cost_R"]:.2f}')
            if cost_parts:
                parts.append("C:" + ",".join(cost_parts))
        if parts:
            ax.text(x + 0.012, y + 0.012, "\n".join(parts), fontsize=6)

    ax.legend()
    _save(fig, out_path, dpi)


def main():
    args = build_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.input_csv)

    # 标准化 must_assign_to
    if "must_assign_to" in df.columns:
        df["must_assign_to"] = df["must_assign_to"].replace({np.nan: "None"})

    plot_label_map(df, os.path.join(args.output_dir, "01_label_map.png"), args.dpi)
    plot_allowed_arms_map(df, os.path.join(args.output_dir, "02_allowed_arms_map.png"), args.dpi)
    plot_safe_free_counts(df, os.path.join(args.output_dir, "03_safe_free_counts.png"), args.dpi)
    plot_optimizer_cost_map(df, os.path.join(args.output_dir, "04_optimizer_costs.png"), args.dpi)
    plot_compact_overview(
        df,
        os.path.join(args.output_dir, "05_compact_overview.png"),
        args.dpi,
        annotate_counts=args.annotate_counts,
        annotate_costs=args.annotate_costs,
    )


if __name__ == "__main__":
    main()
