"""Compare multiple MITS eval summary.json files."""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, List, Mapping, Optional, Tuple


TASK_ROWS = [
    ("Background", "background"),
    ("Recognition", "recognition"),
    ("Counting", "counting"),
    ("Localization", "localization"),
    ("Reasoning", "reasoning"),
]

DEFAULT_LABELS = {
    "base": "Base",
    "traffic_full": "Traffic Full",
    "ours_15_lorasculpt": "Ours 15% + LoRASculpt",
    "ours_15_lorasculpt_adapter": "Ours 15% + LoRASculpt Adapter",
    "ours_15_lorasculpt_merged": "Ours 15% + LoRASculpt Merged",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare multiple MITS eval summaries.")
    parser.add_argument(
        "--summary",
        action="append",
        required=True,
        help="Summary in label=path form. Can be repeated.",
    )
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--baseline-label", default="base")
    parser.add_argument("--ours-label", default="ours_15_lorasculpt")
    parser.add_argument("--traffic-label", default="traffic_full")
    return parser.parse_args()


def _parse_summary_arg(value: str) -> Tuple[str, str]:
    if "=" not in value:
        raise ValueError("--summary must use label=path form.")
    label, path = value.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise ValueError("--summary must use non-empty label=path form.")
    return label, path


def _load(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _display_label(label: str) -> str:
    return DEFAULT_LABELS.get(label, label)


def _task_score(summary: Mapping[str, Any], task: str) -> Optional[float]:
    value = summary.get("tasks", {}).get(task, {}).get("score")
    if value is None:
        return None
    return float(value)


def _metric_score(summary: Mapping[str, Any], task: Optional[str]) -> Optional[float]:
    if task is None:
        value = summary.get("mits_avg")
        if value is None and summary.get("evaluation_type") == "external":
            value = summary.get("external_avg")
        if value is None:
            return None
        return float(value)
    return _task_score(summary, task)


def _delta(left: Optional[float], right: Optional[float]) -> Optional[float]:
    if left is None or right is None:
        return None
    return left - right


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return "incomplete"
    return f"{value * 100:.2f}"


def _overall_metric_name(summaries: Mapping[str, Mapping[str, Any]]) -> str:
    if summaries and all(summary.get("evaluation_type") == "external" for summary in summaries.values()):
        return "External Avg"
    return "MITS Avg"


def _rows(
    labels: List[str],
    summaries: Mapping[str, Mapping[str, Any]],
    ours_label: str,
    baseline_label: str,
    traffic_label: str,
) -> List[Dict[str, str]]:
    metrics: List[Tuple[str, Optional[str]]] = list(TASK_ROWS)
    metrics.append((_overall_metric_name(summaries), None))

    rows: List[Dict[str, str]] = []
    for metric_name, task in metrics:
        row: Dict[str, str] = {"Metric": metric_name}
        metric_scores = {
            label: _metric_score(summaries[label], task)
            for label in labels
        }
        for label in labels:
            row[_display_label(label)] = _fmt(metric_scores[label])

        ours_score = metric_scores.get(ours_label)
        base_score = metric_scores.get(baseline_label)
        traffic_score = metric_scores.get(traffic_label)
        row["Ours - Base"] = _fmt(_delta(ours_score, base_score))
        row["Ours - Traffic Full"] = _fmt(_delta(ours_score, traffic_score))
        rows.append(row)
    return rows


def _fieldnames(labels: List[str]) -> List[str]:
    return ["Metric"] + [_display_label(label) for label in labels] + ["Ours - Base", "Ours - Traffic Full"]


def _write_md(rows: List[Dict[str, str]], labels: List[str], output_path: str) -> None:
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    fields = _fieldnames(labels)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("# Base Qwen vs Traffic Full vs Ours 15% + LoRASculpt\n\n")
        handle.write("Scores are percentages. `incomplete` means judge scores are missing for open-ended tasks.\n\n")
        handle.write("| " + " | ".join(fields) + " |\n")
        handle.write("|" + "|".join(["---"] + ["---:"] * (len(fields) - 1)) + "|\n")
        for row in rows:
            handle.write("| " + " | ".join(row[field] for field in fields) + " |\n")


def _write_csv(rows: List[Dict[str, str]], labels: List[str], output_path: str) -> None:
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fields = _fieldnames(labels)
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    label_paths = [_parse_summary_arg(value) for value in args.summary]
    labels = [label for label, _path in label_paths]
    summaries = {label: _load(path) for label, path in label_paths}
    rows = _rows(
        labels=labels,
        summaries=summaries,
        ours_label=args.ours_label,
        baseline_label=args.baseline_label,
        traffic_label=args.traffic_label,
    )
    _write_md(rows, labels, args.output_md)
    _write_csv(rows, labels, args.output_csv)
    print(f"Wrote {args.output_md}")
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
