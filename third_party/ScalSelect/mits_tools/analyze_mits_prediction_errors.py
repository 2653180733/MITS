"""Compare MITS per-sample scores across models and export hard error slices."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from mits_pipeline.eval_utils import read_json_or_jsonl, write_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze MITS prediction errors across scored model outputs.")
    parser.add_argument(
        "--scores",
        action="append",
        required=True,
        help="Model score JSONL as label=path. Use per_sample_scores.jsonl files.",
    )
    parser.add_argument("--focus-model", required=True, help="Label to analyze as the candidate model.")
    parser.add_argument("--reference-model", default=None, help="Optional strong reference model, e.g. traffic_full.")
    parser.add_argument("--baseline-model", default=None, help="Optional baseline model, e.g. ours_15_lorasculpt_merged.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-examples", type=int, default=100)
    parser.add_argument("--task", action="append", default=[], help="Restrict exported examples to task(s).")
    return parser.parse_args()


def _parse_labeled_path(value: str) -> Tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"Expected label=path, got: {value}")
    label, path = value.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise ValueError(f"Expected non-empty label=path, got: {value}")
    return label, path


def _score_value(record: Mapping[str, Any]) -> Optional[float]:
    value = record.get("score")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_scores(paths: Sequence[str]) -> Tuple[Dict[str, Dict[str, Dict[str, Any]]], Dict[str, Dict[str, Any]]]:
    by_model: Dict[str, Dict[str, Dict[str, Any]]] = {}
    metadata: Dict[str, Dict[str, Any]] = {}
    for value in paths:
        label, path = _parse_labeled_path(value)
        records: Dict[str, Dict[str, Any]] = {}
        for record in read_json_or_jsonl(path):
            sample_id = str(record.get("id"))
            if not sample_id:
                continue
            item = dict(record)
            item["_score_float"] = _score_value(item)
            records[sample_id] = item
        by_model[label] = records
        metadata[label] = {"path": path, "records": len(records)}
    return by_model, metadata


def _is_good(score: Optional[float], threshold: float = 0.999) -> bool:
    return score is not None and score >= threshold


def _is_bad(score: Optional[float], threshold: float = 0.999) -> bool:
    return score is not None and score < threshold


def _scene(record: Mapping[str, Any]) -> str:
    return str(record.get("scene") or "unknown")


def _task(record: Mapping[str, Any]) -> str:
    return str(record.get("task") or "unknown")


def _make_joined_rows(by_model: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> List[Dict[str, Any]]:
    ids = sorted(set.intersection(*(set(records.keys()) for records in by_model.values()))) if by_model else []
    labels = list(by_model.keys())
    rows: List[Dict[str, Any]] = []
    for sample_id in ids:
        first = dict(by_model[labels[0]][sample_id])
        row = {
            "id": sample_id,
            "task": _task(first),
            "scene": _scene(first),
            "field": first.get("field"),
            "question": first.get("question"),
            "answer": first.get("answer"),
            "image": first.get("image"),
        }
        for label in labels:
            record = by_model[label][sample_id]
            row[f"{label}_score"] = record.get("_score_float")
            row[f"{label}_prediction"] = record.get("prediction")
            row[f"{label}_score_source"] = record.get("score_source")
            row[f"{label}_score_detail"] = record.get("score_detail")
        rows.append(row)
    return rows


def _mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _aggregate(rows: Sequence[Mapping[str, Any]], labels: Sequence[str], group_key: str) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_key) or "unknown")].append(row)

    output = []
    for group, group_rows in sorted(grouped.items()):
        item: Dict[str, Any] = {group_key: group, "n": len(group_rows)}
        for label in labels:
            scored = [float(row[f"{label}_score"]) for row in group_rows if row.get(f"{label}_score") is not None]
            item[f"{label}_avg"] = _mean(scored)
            item[f"{label}_scored_n"] = len(scored)
        output.append(item)
    return output


def _write_csv(rows: Sequence[Mapping[str, Any]], path: str) -> None:
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    preferred = ["task", "scene", "n", "id", "field", "question", "answer", "image"]
    fieldnames = [key for key in preferred if key in fieldnames] + [key for key in fieldnames if key not in preferred]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _filter_tasks(rows: Iterable[Mapping[str, Any]], tasks: Sequence[str]) -> List[Mapping[str, Any]]:
    if not tasks:
        return list(rows)
    allowed = {task.lower() for task in tasks}
    return [row for row in rows if str(row.get("task") or "").lower() in allowed]


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    by_model, metadata = _load_scores(args.scores)
    labels = list(by_model.keys())
    if args.focus_model not in by_model:
        raise ValueError(f"--focus-model not found in --scores labels: {args.focus_model}")
    if args.reference_model and args.reference_model not in by_model:
        raise ValueError(f"--reference-model not found in --scores labels: {args.reference_model}")
    if args.baseline_model and args.baseline_model not in by_model:
        raise ValueError(f"--baseline-model not found in --scores labels: {args.baseline_model}")

    rows = _make_joined_rows(by_model)
    focus = args.focus_model
    reference = args.reference_model
    baseline = args.baseline_model

    task_summary = _aggregate(rows, labels, "task")
    scene_summary = _aggregate(rows, labels, "scene")
    _write_csv(task_summary, os.path.join(args.output_dir, "score_by_task.csv"))
    _write_csv(scene_summary, os.path.join(args.output_dir, "score_by_scene.csv"))

    hard_rows = []
    regressions = []
    improvements = []
    for row in rows:
        focus_score = row.get(f"{focus}_score")
        reference_score = row.get(f"{reference}_score") if reference else None
        baseline_score = row.get(f"{baseline}_score") if baseline else None
        if reference and _is_bad(focus_score) and _is_good(reference_score):
            hard_rows.append(row)
        if baseline and _is_bad(focus_score) and _is_good(baseline_score):
            regressions.append(row)
        if baseline and _is_good(focus_score) and _is_bad(baseline_score):
            improvements.append(row)

    hard_rows = _filter_tasks(hard_rows, args.task)
    regressions = _filter_tasks(regressions, args.task)
    improvements = _filter_tasks(improvements, args.task)

    hard_rows = hard_rows[: args.max_examples]
    regressions = regressions[: args.max_examples]
    improvements = improvements[: args.max_examples]

    write_jsonl(hard_rows, os.path.join(args.output_dir, "focus_wrong_reference_right.jsonl"))
    write_jsonl(regressions, os.path.join(args.output_dir, "focus_regressions_vs_baseline.jsonl"))
    write_jsonl(improvements, os.path.join(args.output_dir, "focus_improvements_vs_baseline.jsonl"))
    _write_csv(hard_rows, os.path.join(args.output_dir, "focus_wrong_reference_right.csv"))
    _write_csv(regressions, os.path.join(args.output_dir, "focus_regressions_vs_baseline.csv"))
    _write_csv(improvements, os.path.join(args.output_dir, "focus_improvements_vs_baseline.csv"))

    summary = {
        "models": metadata,
        "joined_records": len(rows),
        "focus_model": focus,
        "reference_model": reference,
        "baseline_model": baseline,
        "hard_reference_right": len(hard_rows),
        "regressions_vs_baseline": len(regressions),
        "improvements_vs_baseline": len(improvements),
        "outputs": args.output_dir,
    }
    with open(os.path.join(args.output_dir, "error_analysis_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
