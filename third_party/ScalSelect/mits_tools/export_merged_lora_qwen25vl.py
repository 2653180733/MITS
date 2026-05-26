"""Export a merged Qwen2.5-VL LoRA checkpoint as a standard HF model."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict


DEFAULT_BASE_MODEL = os.environ.get("MODEL_PATH", "/root/autodl-tmp/Qwen/Qwen2.5-VL-7B-Instruct")
DEFAULT_ADAPTER = os.environ.get(
    "OURS_ADAPTER",
    "/root/autodl-tmp/data/train_outputs/mits_15_lorasculpt_full_swanlab/v0-20260523-162150/checkpoint-1540",
)
DEFAULT_OUTPUT = os.environ.get(
    "MERGED_OURS_MODEL",
    "/root/autodl-tmp/data/train_outputs/mits_15_lorasculpt_merged",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a PEFT LoRA adapter into Qwen2.5-VL.")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument(
        "--safe-serialization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save safetensors when supported.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output directory.",
    )
    return parser.parse_args()


def _resolve_dtype(name: str):
    import torch

    if name == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _resolve_device_map(value: str):
    text = str(value or "").strip().lower()
    if text == "auto":
        return "auto"
    if text == "cpu":
        return {"": "cpu"}
    if text.startswith("cuda:"):
        return {"": int(text.split(":", 1)[1])}
    return value


def _ensure_output_dir(path: str, overwrite: bool) -> None:
    if os.path.exists(path) and os.listdir(path) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {path}. "
            "Pass --overwrite or choose a new output path to avoid replacing prior exports."
        )
    os.makedirs(path, exist_ok=True)


def main() -> None:
    args = parse_args()
    _ensure_output_dir(args.output, overwrite=args.overwrite)

    from peft import PeftModel
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    dtype = _resolve_dtype(args.torch_dtype)
    print(f"Loading base model: {args.base_model}", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map=_resolve_device_map(args.device_map),
    )
    print(f"Loading adapter: {args.adapter}", flush=True)
    peft_model = PeftModel.from_pretrained(model, args.adapter)
    print("Merging LoRA adapter into base model...", flush=True)
    merged_model = peft_model.merge_and_unload()
    merged_model.eval()

    print(f"Saving merged model to: {args.output}", flush=True)
    merged_model.save_pretrained(args.output, safe_serialization=args.safe_serialization)

    processor = AutoProcessor.from_pretrained(args.base_model)
    processor.save_pretrained(args.output)

    metadata: Dict[str, Any] = {
        "base_model": args.base_model,
        "adapter": args.adapter,
        "output": args.output,
        "torch_dtype": args.torch_dtype,
        "device_map": args.device_map,
        "safe_serialization": args.safe_serialization,
    }
    with open(os.path.join(args.output, "merge_metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
