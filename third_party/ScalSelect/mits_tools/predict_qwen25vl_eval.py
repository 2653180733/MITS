"""Run Qwen2.5-VL predictions for MITS eval JSONL files."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Iterable, List, Optional


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

DEFAULT_MODEL_PATH = os.environ.get("MODEL_PATH", "/root/autodl-tmp/Qwen/Qwen2.5-VL-7B-Instruct")

from mits_pipeline.eval_utils import read_json_or_jsonl, resolve_image_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict MITS eval answers with Qwen2.5-VL.")
    parser.add_argument("--eval-jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--model-label", default=None)
    parser.add_argument("--image-root", default=None, help="Optional root for relative image paths.")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--max-pixels", type=int, default=1048576)
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-missing-images", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to an existing output JSONL and skip records whose id is already present.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1,
        help="Print progress every N predictions. The first sample is always logged before generation.",
    )
    parser.add_argument(
        "--skip-shard-check",
        action="store_true",
        help="Skip local HuggingFace safetensors shard existence validation.",
    )
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument(
        "--task-format-instruction",
        default="none",
        choices=["none", "auto"],
        help="Append strict answer-format instructions by task. Default keeps original questions unchanged.",
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


def _missing_hf_shards(model_path: str) -> List[str]:
    if not model_path or not os.path.isdir(model_path):
        return []

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        return []

    with open(index_path, "r", encoding="utf-8") as handle:
        index = json.load(handle)
    weight_map = index.get("weight_map", {})
    expected = sorted(set(weight_map.values()))
    return [name for name in expected if not os.path.exists(os.path.join(model_path, name))]


def _validate_hf_shards(model_path: str) -> None:
    missing = _missing_hf_shards(model_path)
    if not missing:
        return
    preview = ", ".join(missing[:8])
    if len(missing) > 8:
        preview += f", ... (+{len(missing) - 8} more)"
    raise FileNotFoundError(
        "The HuggingFace model directory is missing safetensors shard files "
        f"referenced by model.safetensors.index.json: {preview}. "
        "For Traffic Full, run this on AutoDL where the full checkpoint shards exist."
    )


def _validate_local_model_path(model_path: str) -> None:
    if not model_path or not os.path.isabs(model_path):
        return
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Local model path does not exist: {model_path}. "
            "If this is Ours merged, rerun export_merged_lora_qwen25vl.py or set MERGED_OURS_MODEL "
            "to the actual merged model directory."
        )
    if not os.path.isdir(model_path):
        raise NotADirectoryError(f"Local model path is not a directory: {model_path}")
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Local model directory is missing config.json: {model_path}. "
            "This does not look like a complete HuggingFace model export."
        )


def _load_model_and_processor(args: argparse.Namespace):
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    _validate_local_model_path(args.model)
    if not args.skip_shard_check:
        _validate_hf_shards(args.model)
    dtype = _resolve_dtype(args.torch_dtype)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=_resolve_device_map(args.device_map),
    )

    if args.adapter:
        try:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, args.adapter)
        except Exception as exc:  # pragma: no cover - depends on adapter runtime
            raise RuntimeError(
                "Failed to load the LoRA adapter with peft.PeftModel.from_pretrained. "
                "If this checkpoint is not a standard PEFT adapter, export/merge it first "
                "or switch this predictor to the ms-swift adapter loading path."
            ) from exc

    processor_kwargs: Dict[str, Any] = {"max_pixels": args.max_pixels}
    if args.min_pixels is not None:
        processor_kwargs["min_pixels"] = args.min_pixels
    processor = AutoProcessor.from_pretrained(args.model, **processor_kwargs)
    if hasattr(processor, "image_processor"):
        if hasattr(processor.image_processor, "max_pixels"):
            processor.image_processor.max_pixels = args.max_pixels
        if args.min_pixels is not None and hasattr(processor.image_processor, "min_pixels"):
            processor.image_processor.min_pixels = args.min_pixels

    model.eval()
    torch.set_grad_enabled(False)
    return model, processor


def _device_for_inputs(device_map: str) -> str:
    text = str(device_map or "").strip().lower()
    if text == "cpu":
        return "cpu"
    if text.startswith("cuda:"):
        return text
    if text == "auto":
        return "cuda"
    return "cuda"


def _format_instruction(record: Dict[str, Any], mode: str) -> str:
    if mode == "none":
        return ""
    task = str(record.get("task") or "").lower()
    answer_type = str(record.get("answer_type") or "").lower()
    if task == "counting" or answer_type == "count":
        return "Answer with a single integer only."
    if task == "localization" or answer_type == "bbox":
        return "Answer only with bounding box coordinates in [x1, y1, x2, y2] format. If the target is absent, answer: no object."
    if answer_type == "yesno":
        return "Answer only yes or no."
    return ""


def _predict_one(model, processor, image_path: str, question: str, args: argparse.Namespace) -> Dict[str, Any]:
    import torch
    from PIL import Image

    started_at = time.perf_counter()
    image = Image.open(image_path).convert("RGB")
    messages = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        }
    )
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt")
    input_device = next(model.parameters()).device
    inputs = inputs.to(input_device)

    generation_kwargs: Dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
    }
    if args.do_sample:
        generation_kwargs["temperature"] = max(args.temperature, 1e-6)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_kwargs)
    generated_ids = [
        output[len(input_ids) :]
        for input_ids, output in zip(inputs.input_ids, output_ids)
    ]
    prediction = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0]
    return {
        "prediction": prediction.strip(),
        "sample_time_s": time.perf_counter() - started_at,
        "input_tokens": int(inputs.input_ids.shape[-1]),
        "generated_tokens": int(generated_ids[0].shape[-1]) if generated_ids else 0,
    }


def _iter_limited(path: str, limit: Optional[int]) -> Iterable[Dict[str, Any]]:
    for index, record in enumerate(read_json_or_jsonl(path)):
        if limit is not None and index >= limit:
            break
        yield dict(record)


def _completed_ids(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    completed: set[str] = set()
    for record in read_json_or_jsonl(path):
        sample_id = record.get("id")
        if sample_id is not None:
            completed.add(str(sample_id))
    return completed


def _open_output(path: str, resume: bool):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    mode = "a" if resume else "w"
    return open(path, mode, encoding="utf-8", buffering=1)


def main() -> None:
    args = parse_args()
    model_label = args.model_label or ("ours" if args.adapter else "base")
    model, processor = _load_model_and_processor(args)

    completed = _completed_ids(args.output) if args.resume else set()
    total_written = 0
    skipped = 0
    started_at = time.time()
    progress_every = max(1, int(args.progress_every or 1))

    print(
        f"Starting prediction: model_label={model_label}, eval_jsonl={args.eval_jsonl}, "
        f"output={args.output}, limit={args.limit}, resume={args.resume}, "
        f"completed={len(completed)}",
        flush=True,
    )

    handle = _open_output(args.output, resume=args.resume)
    for index, record in enumerate(_iter_limited(args.eval_jsonl, args.limit), start=1):
        sample_id = str(record.get("id", index))
        if sample_id in completed:
            skipped += 1
            if skipped == 1 or skipped % progress_every == 0:
                print(f"Skipped {skipped} completed samples; latest id={sample_id}", flush=True)
            continue

        image_path = str(record.get("image") or "")
        if not os.path.isabs(image_path):
            image_path = resolve_image_path(image_path, image_root=args.image_root)
        if not os.path.exists(image_path):
            message = f"Missing image for sample {record.get('id')}: {image_path}"
            if args.skip_missing_images:
                print(f"Warning: {message}; skipped.")
                continue
            raise FileNotFoundError(message)

        if total_written == 0 or (total_written + 1) % progress_every == 0:
            print(
                f"Predicting sample_index={index}, written={total_written}, "
                f"id={sample_id}, image={image_path}",
                flush=True,
            )
        sample_started_at = time.time()
        question = str(record.get("question") or "")
        instruction = _format_instruction(record, args.task_format_instruction)
        if instruction:
            question = f"{question}\n\n{instruction}"
        prediction_info = _predict_one(
            model=model,
            processor=processor,
            image_path=image_path,
            question=question,
            args=args,
        )
        item = dict(record)
        item.update(
            {
                "prediction": prediction_info["prediction"],
                "sample_time_s": prediction_info["sample_time_s"],
                "input_tokens": prediction_info["input_tokens"],
                "generated_tokens": prediction_info["generated_tokens"],
                "model_label": model_label,
                "model_path": args.model,
                "adapter_path": args.adapter,
            }
        )
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        handle.flush()
        total_written += 1
        if total_written == 1 or total_written % progress_every == 0:
            elapsed = time.time() - started_at
            speed = elapsed / max(1, total_written)
            print(
                f"Predicted written={total_written}, source_index={index}, "
                f"id={sample_id}, sample_time={prediction_info['sample_time_s']:.2f}s, "
                f"avg_time={speed:.2f}s, generated_tokens={prediction_info['generated_tokens']}",
                flush=True,
            )

    handle.close()
    print(
        f"Finished prediction: wrote={total_written}, skipped={skipped}, "
        f"output={args.output}, elapsed={time.time() - started_at:.2f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
