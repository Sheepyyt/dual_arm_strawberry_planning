#!/usr/bin/env python3
"""
build_point_motion_table_from_per_ik.py

把 per-IK 批量 OMPL 结果整理成“点级”表，供后续优化器直接读取。

输入（默认）:
  ompl/results/batch_plan_roi_per_ik/motion_roi_table_selected_points_per_ik.pkl

输出（默认）:
  ompl/results/point_motion_table/point_motion_table_selected_points.pkl
  ompl/results/point_motion_table/point_motion_table_selected_points_summary.csv
  ompl/results/point_motion_table/point_motion_table_summary.txt

规则（按用户澄清后的版本）:
1) 若某点至少存在一条 safe 结果（任一臂）:
   - 该点标记为 parallel
   - 优化器只允许使用 safe 候选
   - 若仅 L 有 safe: must_assign_to=L, allowed_arms=[L]
   - 若仅 R 有 safe: must_assign_to=R, allowed_arms=[R]
   - 若 L/R 都有 safe: must_assign_to=None, allowed_arms=[L,R]

2) 若某点两臂都没有 safe，但存在 free:
   - 上半区 -> serial_upper
   - 下半区 -> serial_lower
   - 优化器只允许使用 free 候选
   - allowed_arms 为实际有 free 候选的臂

3) 若某点两臂都没有 safe，也没有 free:
   - discard=True
   - 不进入后续优化器
"""
import os
import csv
import math
import pickle
import argparse
from typing import Dict, List, Any


def parse_args():
    p = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    default_in = os.path.join(here, "results", "batch_plan_roi_per_ik", "motion_roi_table_selected_points_per_ik.pkl")
    default_out_dir = os.path.join(here, "results", "point_motion_table")
    p.add_argument("--input-pkl", type=str, default=default_in)
    p.add_argument("--output-dir", type=str, default=default_out_dir)
    return p.parse_args()


def parse_key_xyz(key: str):
    x, y, z = key.split("_")
    return float(x), float(y), float(z)


def half_from_y(y: float) -> str:
    return "upper" if y >= 0.25 else "lower"


def successful_candidates(arm_payload: Dict[str, Any], mode: str) -> List[Dict[str, Any]]:
    out = []
    for rec in arm_payload.get("per_ik_results", []):
        if not rec.get("success", False):
            continue
        if rec.get("used_mode") != mode:
            continue
        out.append(rec)
    return out


def simplify_candidate(rec: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ik_index": int(rec["ik_index"]),
        "best_cost": None if rec.get("best_cost") is None else float(rec["best_cost"]),
        "goal_q": rec.get("goal_q"),
        "best_path": rec.get("best_path"),
        "used_mode": rec.get("used_mode"),
        "figure_relpath": rec.get("figure_relpath", ""),
        "is_best_success_for_key_arm": bool(rec.get("is_best_success_for_key_arm", False)),
    }


def best_cost(cands: List[Dict[str, Any]]):
    vals = [c["best_cost"] for c in cands if c.get("best_cost") is not None]
    return None if not vals else float(min(vals))


def build_point_record(key: str, arms_payload: Dict[str, Any]) -> Dict[str, Any]:
    x, y, z = parse_key_xyz(key)
    half = half_from_y(y)

    left_payload = arms_payload.get("L", {})
    right_payload = arms_payload.get("R", {})

    safe_L = [simplify_candidate(r) for r in successful_candidates(left_payload, "safe")]
    safe_R = [simplify_candidate(r) for r in successful_candidates(right_payload, "safe")]
    free_L = [simplify_candidate(r) for r in successful_candidates(left_payload, "free")]
    free_R = [simplify_candidate(r) for r in successful_candidates(right_payload, "free")]

    has_safe_L = len(safe_L) > 0
    has_safe_R = len(safe_R) > 0
    has_free_L = len(free_L) > 0
    has_free_R = len(free_R) > 0

    point = {
        "key": key,
        "x": x,
        "y": y,
        "z": z,
        "half": half,
        "safe_count_L": len(safe_L),
        "safe_count_R": len(safe_R),
        "free_count_L": len(free_L),
        "free_count_R": len(free_R),
        "safe_candidates_L": safe_L,
        "safe_candidates_R": safe_R,
        "free_candidates_L": free_L,
        "free_candidates_R": free_R,
        "discard": False,
        "label": None,
        "allowed_arms": [],
        "must_assign_to": None,
        "optimizer_candidates_L": [],
        "optimizer_candidates_R": [],
        "optimizer_best_cost_L": None,
        "optimizer_best_cost_R": None,
        "case": None,
    }

    # 1) any safe => parallel, and only safe are passed to optimizer
    if has_safe_L or has_safe_R:
        point["label"] = "parallel"
        if has_safe_L and has_safe_R:
            point["allowed_arms"] = ["L", "R"]
            point["must_assign_to"] = None
            point["case"] = "both_safe"
            point["optimizer_candidates_L"] = safe_L
            point["optimizer_candidates_R"] = safe_R
        elif has_safe_L:
            point["allowed_arms"] = ["L"]
            point["must_assign_to"] = "L"
            point["case"] = "left_safe_only"
            point["optimizer_candidates_L"] = safe_L
            point["optimizer_candidates_R"] = []
        else:
            point["allowed_arms"] = ["R"]
            point["must_assign_to"] = "R"
            point["case"] = "right_safe_only"
            point["optimizer_candidates_L"] = []
            point["optimizer_candidates_R"] = safe_R

    # 2) no safe, but free exists => serial
    elif has_free_L or has_free_R:
        point["label"] = "serial_upper" if half == "upper" else "serial_lower"
        if has_free_L and has_free_R:
            point["allowed_arms"] = ["L", "R"]
            point["must_assign_to"] = None
            point["case"] = "both_free_no_safe"
            point["optimizer_candidates_L"] = free_L
            point["optimizer_candidates_R"] = free_R
        elif has_free_L:
            point["allowed_arms"] = ["L"]
            point["must_assign_to"] = "L"
            point["case"] = "left_free_only_no_safe"
            point["optimizer_candidates_L"] = free_L
            point["optimizer_candidates_R"] = []
        else:
            point["allowed_arms"] = ["R"]
            point["must_assign_to"] = "R"
            point["case"] = "right_free_only_no_safe"
            point["optimizer_candidates_L"] = []
            point["optimizer_candidates_R"] = free_R

    # 3) no result at all
    else:
        point["discard"] = True
        point["label"] = "discard"
        point["allowed_arms"] = []
        point["must_assign_to"] = None
        point["case"] = "no_result"

    point["optimizer_best_cost_L"] = best_cost(point["optimizer_candidates_L"])
    point["optimizer_best_cost_R"] = best_cost(point["optimizer_candidates_R"])
    return point


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.input_pkl, "rb") as f:
        payload = pickle.load(f)

    data = payload["data"]
    point_records = []
    for key in sorted(data.keys()):
        point_records.append(build_point_record(key, data[key]))

    out_payload = {
        "meta": {
            "source_pkl": args.input_pkl,
            "requested_mode": payload.get("meta", {}).get("requested_mode"),
            "planner": payload.get("meta", {}).get("planner"),
            "rule_version": "safe-first-point-level-labeling",
        },
        "points": point_records,
    }

    out_pkl = os.path.join(args.output_dir, "point_motion_table_selected_points.pkl")
    out_csv = os.path.join(args.output_dir, "point_motion_table_selected_points_summary.csv")
    out_txt = os.path.join(args.output_dir, "point_motion_table_summary.txt")

    with open(out_pkl, "wb") as f:
        pickle.dump(out_payload, f, protocol=4)

    fieldnames = [
        "key", "x", "y", "z", "half",
        "label", "case", "discard",
        "allowed_arms", "must_assign_to",
        "safe_count_L", "safe_count_R",
        "free_count_L", "free_count_R",
        "optimizer_best_cost_L", "optimizer_best_cost_R",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in point_records:
            row = {k: p[k] for k in fieldnames}
            row["allowed_arms"] = ",".join(p["allowed_arms"])
            writer.writerow(row)

    n_total = len(point_records)
    n_discard = sum(int(p["discard"]) for p in point_records)
    n_parallel = sum(int(p["label"] == "parallel") for p in point_records)
    n_serial_upper = sum(int(p["label"] == "serial_upper") for p in point_records)
    n_serial_lower = sum(int(p["label"] == "serial_lower") for p in point_records)

    case_counts = {}
    for p in point_records:
        case_counts[p["case"]] = case_counts.get(p["case"], 0) + 1

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"source_pkl: {args.input_pkl}\n")
        f.write(f"total points: {n_total}\n")
        f.write(f"parallel: {n_parallel}\n")
        f.write(f"serial_upper: {n_serial_upper}\n")
        f.write(f"serial_lower: {n_serial_lower}\n")
        f.write(f"discard: {n_discard}\n")
        f.write("case counts:\n")
        for k in sorted(case_counts.keys()):
            f.write(f"  {k}: {case_counts[k]}\n")

    print(f"Saved point-level pkl to {out_pkl}")
    print(f"Saved point-level csv to {out_csv}")
    print(f"Saved summary to {out_txt}")


if __name__ == "__main__":
    main()
