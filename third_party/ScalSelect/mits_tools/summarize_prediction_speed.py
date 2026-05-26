"""Summarize per-sample prediction latency and token counts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from typing import Any, Dict, List, Mapping, Optional, Tuple


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from mits_pipeline.eval_utils import read_json_or_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize prediction speed JSONL files.")
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        help="Prediction file in label=path form. Can be repeated.",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", default=None)
    return parser.parse_args()


def _parse_label_path(value: str) -> Tuple[str, str]:
    if "=" not in value:
        raise ValueError("--prediction must use label=path form.")
    label, path = value.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise ValueError("--prediction must use non-empty label=path form.")
    return label, path


def _percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - pos) + ordered[upper] * (pos - lower)


def _avg(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _summarize(label: str, path: str) -> Dict[str, Any]:
    sample_times: List[float] = []
    input_tokens: List[float] = []
    generated_tokens: List[float] = []
    model_path = None
    adapter_path = None
    total = 0

    for record in read_json_or_jsonl(path):
        total += 1
        if model_path is None:
            model_path = record.get("model_path")
        if adapter_path is None:
            adapter_path = record.get("adapter_path")
        if record.get("sample_time_s") is not None:
            sample_times.append(float(record["sample_time_s"]))
        if record.get("input_tokens") is not None:
            input_tokens.append(float(record["input_tokens"]))
        if record.get("generated_tokens") is not None:
            generated_tokens.append(float(record["generated_tokens"]))

    total_time = sum(sample_times)
    return {
        "label": label,
        "path": path,
        "model_path": model_path,
        "adapter_path": adapter_path,
        "total": total,
        "timed_total": len(sample_times),
        "avg_s_per_qa": _avg(sample_times),
        "p50_s": _percentile(sample_times, 0.50),
        "p90_s": _percentile(sample_times, 0.90),
        "p95_s": _percentile(sample_times, 0.95),
        "throughput_qa_per_s": len(sample_times) / total_time if total_time > 0 else None,
        "avg_input_tokens": _avg(input_tokens),
        "avg_generated_tokens": _avg(generated_tokens),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _write_json(rows: List[Mapping[str, Any]], path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"models": rows}, handle, ensure_ascii=False, indent=2)


def _write_csv(rows: List[Mapping[str, Any]], path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fields = [
        "label",
        "total",
        "timed_total",
        "avg_s_per_qa",
        "p50_s",
        "p90_s",
        "p95_s",
        "throughput_qa_per_s",
        "avg_input_tokens",
        "avg_generated_tokens",
        "model_path",
        "adapter_path",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _write_md(rows: List[Mapping[str, Any]], path: Optional[str]) -> None:
    if not path:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fields = [
        "label",
        "total",
        "avg_s_per_qa",
        "p50_s",
        "p90_s",
        "p95_s",
        "throughput_qa_per_s",
        "avg_generated_tokens",
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# Prediction Speed Summary\n\n")
        handle.write("| " + " | ".join(fields) + " |\n")
        handle.write("|" + "|".join(["---"] + ["---:"] * (len(fields) - 1)) + "|\n")
        for row in rows:
            handle.write("| " + " | ".join(_fmt(row.get(field)) for field in fields) + " |\n")


def main() -> None:
    args = parse_args()
    rows = [_summarize(*_parse_label_path(value)) for value in args.prediction]
    _write_json(rows, args.output_json)
    _write_csv(rows, args.output_csv)
    _write_md(rows, args.output_md)
    print(json.dumps({"models": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
