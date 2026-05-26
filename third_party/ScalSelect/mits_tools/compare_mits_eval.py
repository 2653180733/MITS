"""Compare base and LoRASculpt MITS eval summaries."""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, Mapping, Optional


TASK_ROWS = [
    ("Background", "background"),
    ("Recognition", "recognition"),
    ("Counting", "counting"),
    ("Localization", "localization"),
    ("Reasoning", "reasoning"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare base and ours summary.json files.")
    parser.add_argument("--base-summary", required=True)
    parser.add_argument("--ours-summary", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def _load(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _task_score(summary: Mapping[str, Any], task: str) -> Optional[float]:
    value = summary.get("tasks", {}).get(task, {}).get("score")
    if value is None:
        return None
    return float(value)


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return "incomplete"
    return f"{value * 100:.2f}"


def _delta(base: Optional[float], ours: Optional[float]) -> Optional[float]:
    if base is None or ours is None:
        return None
    return ours - base


def _row(label: str, base: Optional[float], ours: Optional[float]) -> Dict[str, str]:
    delta = _delta(base, ours)
    return {
        "Metric": label,
        "Base Qwen": _fmt(base),
        "Ours 15% + LoRASculpt": _fmt(ours),
        "Delta": _fmt(delta) if delta is not None else "incomplete",
    }


def _write_md(rows: list[Dict[str, str]], base_summary: Mapping[str, Any], ours_summary: Mapping[str, Any], path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# Base Qwen vs Ours 15% + LoRASculpt\n\n")
        handle.write(f"- Base summary: `{base_summary.get('predictions', '')}`\n")
        handle.write(f"- Ours summary: `{ours_summary.get('predictions', '')}`\n")
        handle.write("- Scores are percentages. `incomplete` means background/reasoning judge scores are missing.\n\n")
        handle.write("| Metric | Base Qwen | Ours 15% + LoRASculpt | Delta |\n")
        handle.write("|---|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                f"| {row['Metric']} | {row['Base Qwen']} | {row['Ours 15% + LoRASculpt']} | {row['Delta']} |\n"
            )


def _write_csv(rows: list[Dict[str, str]], path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Metric", "Base Qwen", "Ours 15% + LoRASculpt", "Delta"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    base_summary = _load(args.base_summary)
    ours_summary = _load(args.ours_summary)

    rows = [
        _row(label, _task_score(base_summary, task), _task_score(ours_summary, task))
        for label, task in TASK_ROWS
    ]
    rows.append(_row("MITS Avg", base_summary.get("mits_avg"), ours_summary.get("mits_avg")))

    _write_md(rows, base_summary, ours_summary, args.output_md)
    _write_csv(rows, args.output_csv)
    print(f"Wrote {args.output_md}")
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
