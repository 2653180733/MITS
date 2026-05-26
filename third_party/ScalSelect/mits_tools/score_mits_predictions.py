"""Score MITS eval predictions and prepare judge files for open-ended tasks."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from mits_pipeline.eval_utils import (  # noqa: E402
    load_judge_scores,
    read_json_or_jsonl,
    score_prediction,
    summarize_scores,
    write_jsonl,
    write_summary_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score MITS prediction JSONL.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--judge-scores", default=None, help="Optional JSONL with {id, score}.")
    parser.add_argument("--external", action="store_true", help="Report external_avg instead of MITS Avg.")
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--summary-csv", default=None)
    parser.add_argument("--per-sample-output", default=None)
    parser.add_argument("--judge-output", default=None)
    return parser.parse_args()


def _judge_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": record.get("id"),
        "task": record.get("task"),
        "field": record.get("field"),
        "image": record.get("image"),
        "question": record.get("question"),
        "reference_answer": record.get("answer"),
        "prediction": record.get("prediction"),
        "score": None,
        "judge_instruction": "Score the prediction against the reference answer from 0 to 1.",
    }


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    summary_json = args.summary_json or os.path.join(args.output_dir, "summary.json")
    summary_csv = args.summary_csv or os.path.join(args.output_dir, "summary.csv")
    per_sample_output = args.per_sample_output or os.path.join(args.output_dir, "per_sample_scores.jsonl")
    judge_output = args.judge_output or os.path.join(args.output_dir, "judge_background_reasoning.jsonl")

    judge_scores = load_judge_scores(args.judge_scores)
    scored_records: List[Dict[str, Any]] = []
    judge_needed: List[Dict[str, Any]] = []

    for record in read_json_or_jsonl(args.predictions):
        item = dict(record)
        judge_score = judge_scores.get(str(item.get("id")))
        score_info = score_prediction(item, judge_score=judge_score)
        item.update(score_info)
        scored_records.append(item)
        if item.get("needs_judge"):
            judge_needed.append(_judge_record(item))

    summary = summarize_scores(scored_records, external=args.external)
    summary.update(
        {
            "predictions": args.predictions,
            "model_label": scored_records[0].get("model_label") if scored_records else None,
            "model_path": scored_records[0].get("model_path") if scored_records else None,
            "adapter_path": scored_records[0].get("adapter_path") if scored_records else None,
            "judge_scores": args.judge_scores,
            "outputs": {
                "summary_json": summary_json,
                "summary_csv": summary_csv,
                "per_sample_scores": per_sample_output,
                "judge_background_reasoning": judge_output,
            },
        }
    )

    write_jsonl(scored_records, per_sample_output)
    write_jsonl(judge_needed, judge_output)
    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    write_summary_csv(summary, summary_csv)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
