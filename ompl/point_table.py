#!/usr/bin/env python3
"""
point_table.py

将 per-IK 批量 OMPL 结果整理成点级 point table，供优化器直接读取，
并自动生成可视化。

默认输入:
  ompl/results/batch/per_ik.pkl

默认输出目录:
  ompl/results/point_table/
    - point_table.pkl
    - point_table.csv
    - summary.txt
    - visualizations/*.png

规则（按当前项目最终确认版本）:
1) 只要某点任一臂存在 safe 结果 -> 该点为 parallel，且优化器只允许使用 safe 候选。
2) 若两臂都没有 safe，但存在 free -> 上半区为 serial_upper，下半区为 serial_lower，且优化器只允许使用 free 候选。
3) 若两臂既无 safe 也无 free -> discard。
"""
import os
import csv
import pickle
import argparse
from typing import Dict, List, Any

import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser()
    p.add_argument('--input-pkl', type=str, default=os.path.join(here, 'results', 'batch', 'per_ik.pkl'))
    p.add_argument('--output-dir', type=str, default=os.path.join(here, 'results', 'point_table'))
    p.add_argument('--force-regenerate', action='store_true')
    p.add_argument('--dpi', type=int, default=220)
    return p.parse_args()


def parse_key_xyz(key: str):
    x, y, z = key.split('_')
    return float(x), float(y), float(z)


def half_from_y(y: float) -> str:
    return 'upper' if y >= 0.25 else 'lower'


def successful_candidates(arm_payload: Dict[str, Any], mode: str) -> List[Dict[str, Any]]:
    out = []
    for rec in arm_payload.get('per_ik_results', []):
        if not rec.get('success', False):
            continue
        if rec.get('used_mode') != mode:
            continue
        out.append(rec)
    return out


def simplify_candidate(rec: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'ik_index': int(rec['ik_index']),
        'best_cost': None if rec.get('best_cost') is None else float(rec['best_cost']),
        'goal_q': rec.get('goal_q'),
        'best_path': rec.get('best_path'),
        'used_mode': rec.get('used_mode'),
        'figure_relpath': rec.get('figure_relpath', ''),
        'is_best_success_for_key_arm': bool(rec.get('is_best_success_for_key_arm', False)),
    }


def best_cost(cands: List[Dict[str, Any]]):
    vals = [c['best_cost'] for c in cands if c.get('best_cost') is not None]
    return None if not vals else float(min(vals))


def build_point_record(key: str, arms_payload: Dict[str, Any]) -> Dict[str, Any]:
    x, y, z = parse_key_xyz(key)
    half = half_from_y(y)

    left_payload = arms_payload.get('L', {})
    right_payload = arms_payload.get('R', {})

    safe_L = [simplify_candidate(r) for r in successful_candidates(left_payload, 'safe')]
    safe_R = [simplify_candidate(r) for r in successful_candidates(right_payload, 'safe')]
    free_L = [simplify_candidate(r) for r in successful_candidates(left_payload, 'free')]
    free_R = [simplify_candidate(r) for r in successful_candidates(right_payload, 'free')]

    has_safe_L = len(safe_L) > 0
    has_safe_R = len(safe_R) > 0
    has_free_L = len(free_L) > 0
    has_free_R = len(free_R) > 0

    point = {
        'key': key,
        'x': x,
        'y': y,
        'z': z,
        'half': half,
        'safe_count_L': len(safe_L),
        'safe_count_R': len(safe_R),
        'free_count_L': len(free_L),
        'free_count_R': len(free_R),
        'safe_candidates_L': safe_L,
        'safe_candidates_R': safe_R,
        'free_candidates_L': free_L,
        'free_candidates_R': free_R,
        'discard': False,
        'label': None,
        'allowed_arms': [],
        'must_assign_to': None,
        'optimizer_candidates_L': [],
        'optimizer_candidates_R': [],
        'optimizer_best_cost_L': None,
        'optimizer_best_cost_R': None,
        'case': None,
    }

    if has_safe_L or has_safe_R:
        point['label'] = 'parallel'
        if has_safe_L and has_safe_R:
            point['allowed_arms'] = ['L', 'R']
            point['must_assign_to'] = None
            point['case'] = 'both_safe'
            point['optimizer_candidates_L'] = safe_L
            point['optimizer_candidates_R'] = safe_R
        elif has_safe_L:
            point['allowed_arms'] = ['L']
            point['must_assign_to'] = 'L'
            point['case'] = 'left_safe_only'
            point['optimizer_candidates_L'] = safe_L
            point['optimizer_candidates_R'] = []
        else:
            point['allowed_arms'] = ['R']
            point['must_assign_to'] = 'R'
            point['case'] = 'right_safe_only'
            point['optimizer_candidates_L'] = []
            point['optimizer_candidates_R'] = safe_R
    elif has_free_L or has_free_R:
        point['label'] = 'serial_upper' if half == 'upper' else 'serial_lower'
        if has_free_L and has_free_R:
            point['allowed_arms'] = ['L', 'R']
            point['must_assign_to'] = None
            point['case'] = 'both_free_no_safe'
            point['optimizer_candidates_L'] = free_L
            point['optimizer_candidates_R'] = free_R
        elif has_free_L:
            point['allowed_arms'] = ['L']
            point['must_assign_to'] = 'L'
            point['case'] = 'left_free_only_no_safe'
            point['optimizer_candidates_L'] = free_L
            point['optimizer_candidates_R'] = []
        else:
            point['allowed_arms'] = ['R']
            point['must_assign_to'] = 'R'
            point['case'] = 'right_free_only_no_safe'
            point['optimizer_candidates_L'] = []
            point['optimizer_candidates_R'] = free_R
    else:
        point['discard'] = True
        point['label'] = 'discard'
        point['allowed_arms'] = []
        point['must_assign_to'] = None
        point['case'] = 'no_result'

    point['optimizer_best_cost_L'] = best_cost(point['optimizer_candidates_L'])
    point['optimizer_best_cost_R'] = best_cost(point['optimizer_candidates_R'])
    return point


def save_tabular(point_records, out_dir):
    out_pkl = os.path.join(out_dir, 'point_table.pkl')
    out_csv = os.path.join(out_dir, 'point_table.csv')
    out_txt = os.path.join(out_dir, 'summary.txt')

    out_payload = {'points': point_records}
    with open(out_pkl, 'wb') as f:
        pickle.dump(out_payload, f, protocol=4)

    fieldnames = [
        'key', 'x', 'y', 'z', 'half',
        'label', 'case', 'discard',
        'allowed_arms', 'must_assign_to',
        'safe_count_L', 'safe_count_R',
        'free_count_L', 'free_count_R',
        'optimizer_best_cost_L', 'optimizer_best_cost_R',
    ]
    with open(out_csv, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in point_records:
            row = {k: p[k] for k in fieldnames}
            row['allowed_arms'] = ','.join(p['allowed_arms'])
            writer.writerow(row)

    n_total = len(point_records)
    n_discard = sum(int(p['discard']) for p in point_records)
    n_parallel = sum(int(p['label'] == 'parallel') for p in point_records)
    n_serial_upper = sum(int(p['label'] == 'serial_upper') for p in point_records)
    n_serial_lower = sum(int(p['label'] == 'serial_lower') for p in point_records)
    case_counts = {}
    for p in point_records:
        case_counts[p['case']] = case_counts.get(p['case'], 0) + 1

    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write(f'total points: {n_total}\n')
        f.write(f'parallel: {n_parallel}\n')
        f.write(f'serial_upper: {n_serial_upper}\n')
        f.write(f'serial_lower: {n_serial_lower}\n')
        f.write(f'discard: {n_discard}\n')
        f.write('case counts:\n')
        for k in sorted(case_counts.keys()):
            f.write(f'  {k}: {case_counts[k]}\n')

    return out_pkl, out_csv, out_txt


def _save(fig, path, dpi):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


def _base_axes(title):
    fig, ax = plt.subplots(figsize=(8.2, 8.0))
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.48, 0.48)
    ax.set_ylim(-0.58, 0.58)
    return fig, ax


def make_visualizations(df: pd.DataFrame, viz_dir: str, dpi: int = 220):
    os.makedirs(viz_dir, exist_ok=True)

    fig, ax = _base_axes('Point labels for optimizer')
    for label, marker in [('parallel', 'o'), ('serial_upper', 's'), ('serial_lower', '^'), ('discard', 'x')]:
        sub = df[df['label'] == label]
        if len(sub) > 0:
            ax.scatter(sub['x'], sub['y'], marker=marker, s=90, label=label)
    for _, r in df.iterrows():
        txt = ''
        if pd.notna(r['must_assign_to']) and str(r['must_assign_to']).strip() not in ('', 'None', 'nan'):
            txt = f"{r['must_assign_to']}"
        if txt:
            ax.text(r['x'] + 0.012, r['y'] + 0.012, txt, fontsize=8)
    ax.legend()
    _save(fig, os.path.join(viz_dir, '01_label_map.png'), dpi)

    fig, ax = _base_axes('Allowed arms / assignment')
    norm = df.copy()
    norm['allowed_arms_norm'] = norm['allowed_arms'].astype(str).str.replace(' ', '', regex=False)
    for tag, marker in [('L', 'o'), ('R', 's'), ('L,R', '^')]:
        sub = norm[norm['allowed_arms_norm'] == tag]
        if len(sub) > 0:
            ax.scatter(sub['x'], sub['y'], marker=marker, s=90, label=f'allowed={tag}')
    for _, r in norm.iterrows():
        label = str(r['label'])
        if label.startswith('serial'):
            ax.text(r['x'] + 0.010, r['y'] - 0.018, 'S', fontsize=8)
        elif label == 'parallel':
            ax.text(r['x'] + 0.010, r['y'] - 0.018, 'P', fontsize=8)
    ax.legend()
    _save(fig, os.path.join(viz_dir, '02_allowed_arms_map.png'), dpi)

    fig, ax = _base_axes('Safe / free counts at each point')
    for label, marker in [('parallel', 'o'), ('serial_upper', 's'), ('serial_lower', '^'), ('discard', 'x')]:
        sub = df[df['label'] == label]
        if len(sub) > 0:
            ax.scatter(sub['x'], sub['y'], marker=marker, s=60, label=label)
    for _, r in df.iterrows():
        x, y = r['x'], r['y']
        ax.text(x + 0.012, y + 0.012, f"L {int(r['safe_count_L'])}/{int(r['free_count_L'])}", fontsize=6)
        ax.text(x + 0.012, y - 0.020, f"R {int(r['safe_count_R'])}/{int(r['free_count_R'])}", fontsize=6)
    ax.legend()
    _save(fig, os.path.join(viz_dir, '03_safe_free_counts.png'), dpi)

    fig, ax = _base_axes('Optimizer single-trip cost (L / R)')
    ax.scatter(df['x'], df['y'], s=50, label='points')
    for _, r in df.iterrows():
        x, y = r['x'], r['y']
        txts = []
        if pd.notna(r['optimizer_best_cost_L']):
            txts.append(f"L {r['optimizer_best_cost_L']:.2f}")
        if pd.notna(r['optimizer_best_cost_R']):
            txts.append(f"R {r['optimizer_best_cost_R']:.2f}")
        if txts:
            ax.text(x + 0.012, y + 0.012, '\n'.join(txts), fontsize=6)
    ax.legend()
    _save(fig, os.path.join(viz_dir, '04_optimizer_costs.png'), dpi)

    fig, ax = _base_axes('Compact overview for optimizer input')
    for label, marker in [('parallel', 'o'), ('serial_upper', 's'), ('serial_lower', '^'), ('discard', 'x')]:
        sub = df[df['label'] == label]
        if len(sub) > 0:
            ax.scatter(sub['x'], sub['y'], marker=marker, s=85, label=label)
    for _, r in df.iterrows():
        x, y = r['x'], r['y']
        parts = []
        ma = str(r['must_assign_to'])
        if ma not in ('', 'None', 'nan'):
            parts.append(f'M={ma}')
        parts.append(f"L {int(r['safe_count_L'])}/{int(r['free_count_L'])}")
        parts.append(f"R {int(r['safe_count_R'])}/{int(r['free_count_R'])}")
        cp = []
        if pd.notna(r['optimizer_best_cost_L']):
            cp.append(f"L {r['optimizer_best_cost_L']:.2f}")
        if pd.notna(r['optimizer_best_cost_R']):
            cp.append(f"R {r['optimizer_best_cost_R']:.2f}")
        if cp:
            parts.append('C:' + ','.join(cp))
        if parts:
            ax.text(x + 0.012, y + 0.012, '\n'.join(parts), fontsize=6)
    ax.legend()
    _save(fig, os.path.join(viz_dir, '05_compact_overview.png'), dpi)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    viz_dir = os.path.join(args.output_dir, 'visualizations')
    out_pkl = os.path.join(args.output_dir, 'point_table.pkl')
    out_csv = os.path.join(args.output_dir, 'point_table.csv')
    out_txt = os.path.join(args.output_dir, 'summary.txt')

    if (not args.force_regenerate) and os.path.exists(out_pkl):
        print(f'Loading existing point table from {out_pkl}')
        with open(out_pkl, 'rb') as f:
            payload = pickle.load(f)
        point_records = payload['points']
    else:
        with open(args.input_pkl, 'rb') as f:
            payload = pickle.load(f)
        data = payload['data']
        point_records = [build_point_record(key, data[key]) for key in sorted(data.keys())]
        save_tabular(point_records, args.output_dir)
        print(f'Saved point table to {out_pkl}')

    if not os.path.exists(out_csv) or not os.path.exists(out_txt) or args.force_regenerate:
        save_tabular(point_records, args.output_dir)

    df = pd.DataFrame([{
        'key': p['key'], 'x': p['x'], 'y': p['y'], 'z': p['z'], 'half': p['half'],
        'label': p['label'], 'case': p['case'], 'discard': p['discard'],
        'allowed_arms': ','.join(p['allowed_arms']), 'must_assign_to': p['must_assign_to'],
        'safe_count_L': p['safe_count_L'], 'safe_count_R': p['safe_count_R'],
        'free_count_L': p['free_count_L'], 'free_count_R': p['free_count_R'],
        'optimizer_best_cost_L': p['optimizer_best_cost_L'], 'optimizer_best_cost_R': p['optimizer_best_cost_R'],
    } for p in point_records])
    make_visualizations(df, viz_dir, args.dpi)


if __name__ == '__main__':
    main()
