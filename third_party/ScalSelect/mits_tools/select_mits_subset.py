"""Select a task-aware MITS subset from an index and optional CUR scores."""

from __future__ import annotations

import argparse
import json
import os
import sys


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

DEFAULT_WORK_DIR = os.environ.get("WORK_DIR", "/root/autodl-tmp/data/outputs/full")
DEFAULT_INDEX_PATH = os.environ.get("MITS_INDEX_PATH", os.path.join(DEFAULT_WORK_DIR, "mits_index.jsonl"))
DEFAULT_CUR_SCORES_PATH = os.environ.get("CUR_SCORES_PATH", os.path.join(DEFAULT_WORK_DIR, "importance_scores.jsonl"))

from mits_pipeline.mits_io import read_jsonl
from mits_pipeline.selection import (
    enrich_records_with_scores,
    load_importance_scores,
    select_records,
    select_records_by_group,
    write_selected_index,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select a task-aware and rarity-aware MITS subset.")
    parser.add_argument(
        "--index",
        default=DEFAULT_INDEX_PATH,
        help=f"Input MITS index JSONL (default: {DEFAULT_INDEX_PATH}).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output selected index JSONL. Default: WORK_DIR/mits_selected_<ratio>.jsonl.",
    )
    parser.add_argument("--cur-scores", default=DEFAULT_CUR_SCORES_PATH, help="ScalSelect importance_scores.jsonl.")
    parser.add_argument("--ratio", type=float, default=15.0, help="Selection ratio. Accepts 0.15 or 15 (default: 15).")
    parser.add_argument(
        "--num-selected",
        type=int,
        default=None,
        help="Absolute selected sample count. In grouped mode this is applied per group.",
    )
    parser.add_argument("--lambda-task", type=float, default=0.2)
    parser.add_argument("--lambda-rare", type=float, default=0.1)
    parser.add_argument("--min-task-fraction", type=float, default=0.05)
    parser.add_argument(
        "--group-by",
        default="scene",
        help="Select independently inside this record field. Use 'none' for global selection.",
    )
    parser.add_argument(
        "--min-per-group",
        type=int,
        default=0,
        help="Minimum selected samples per non-empty group when --group-by is enabled.",
    )
    parser.add_argument(
        "--max-per-group",
        type=int,
        default=None,
        help="Maximum selected samples per group when --group-by is enabled.",
    )
    parser.add_argument(
        "--group-summary-output",
        default=None,
        help="Optional JSONL path for per-group selection counts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output is None:
        label = str(args.num_selected) if args.num_selected is not None else ("%g" % args.ratio)
        label = label.replace(".", "p")
        args.output = os.path.join(DEFAULT_WORK_DIR, f"mits_selected_{label}.jsonl")

    records = list(read_jsonl(args.index))
    cur_scores = None
    if args.cur_scores:
        if os.path.exists(args.cur_scores):
            cur_scores = load_importance_scores(args.cur_scores)
        else:
            print(f"Warning: CUR scores not found at {args.cur_scores}; using metadata-only selection.")
    enriched = enrich_records_with_scores(
        records=records,
        cur_scores=cur_scores,
        lambda_task=args.lambda_task,
        lambda_rare=args.lambda_rare,
    )
    group_by = None if str(args.group_by).lower() in {"", "none", "global"} else args.group_by
    group_summaries = []
    if group_by:
        selected, group_summaries = select_records_by_group(
            records=enriched,
            group_field=group_by,
            ratio=args.ratio,
            num_selected=args.num_selected,
            min_task_fraction=args.min_task_fraction,
            min_per_group=args.min_per_group,
            max_per_group=args.max_per_group,
        )
    else:
        selected = select_records(
            records=enriched,
            ratio=args.ratio,
            num_selected=args.num_selected,
            min_task_fraction=args.min_task_fraction,
        )
    count = write_selected_index(selected, args.output)
    print(f"Wrote {count} selected records to {args.output}")
    if group_summaries:
        print(f"Selection grouped by: {group_by}")
        for summary in group_summaries:
            print(
                "  {group}: selected {selected}/{total} ({ratio:.2%})".format(
                    **summary
                )
            )
    if args.group_summary_output:
        parent = os.path.dirname(args.group_summary_output)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.group_summary_output, "w", encoding="utf-8") as handle:
            for summary in group_summaries:
                handle.write(json.dumps(summary, ensure_ascii=False) + "\n")
        print(f"Wrote group summary to {args.group_summary_output}")


if __name__ == "__main__":
    main()
