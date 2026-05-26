"""Qwen/PEFT friendly LoRASculpt utilities.

This file is added to the local copy under ``D:\\Code\\PyCode\\MITS``. It does
not modify the original reference repository.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import torch
try:
    from transformers import TrainerCallback
except Exception:  # pragma: no cover - transformers exists in training env.
    TrainerCallback = object


DEFAULT_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

DEFAULT_SCULPT_TARGET_KEYWORDS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "mm_projector",
    "multi_modal_projector",
    "visual",
    "merger",
]


def recommended_preserve_ratio(lora_rank: int) -> float:
    """Rank-sculpt co-design defaults for single-card QLoRA."""
    if lora_rank <= 8:
        return 0.15
    if lora_rank <= 16:
        return 0.10
    return 0.05


def is_lora_sculpt_target(name: str, keywords: Optional[Iterable[str]] = None) -> bool:
    keys = list(keywords or DEFAULT_SCULPT_TARGET_KEYWORDS)
    return "lora_" in name and any(key in name for key in keys)


@torch.no_grad()
def sculpt_lora_params(
    model,
    preserve_ratio: float = 0.10,
    target_keywords: Optional[Iterable[str]] = None,
) -> Dict[str, int]:
    """Keep top-magnitude LoRA weights and zero the rest in selected modules."""
    if preserve_ratio <= 0 or preserve_ratio > 1:
        raise ValueError("preserve_ratio must be in (0, 1].")

    kept: Dict[str, int] = {}
    for name, param in model.named_parameters():
        if not is_lora_sculpt_target(name, keywords=target_keywords):
            continue
        if not getattr(param, "requires_grad", False):
            continue

        values = param.data
        flat_abs = values.abs().flatten()
        top_k = max(1, int(flat_abs.numel() * preserve_ratio))
        threshold = flat_abs.topk(top_k).values.min()
        mask = values.abs() >= threshold
        values.mul_(mask)
        kept[name] = int(mask.sum().item())
    return kept


class PeriodicQwenLoRASculptCallback(TrainerCallback):
    """HuggingFace Trainer callback for periodic LoRA sculpting."""

    def __init__(
        self,
        sculpt_interval: int = 300,
        preserve_ratio: float = 0.10,
        target_keywords: Optional[List[str]] = None,
    ) -> None:
        if sculpt_interval <= 0:
            raise ValueError("sculpt_interval must be positive.")
        self.sculpt_interval = sculpt_interval
        self.preserve_ratio = preserve_ratio
        self.target_keywords = target_keywords

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return control
        if state.global_step <= 0:
            return control
        if state.global_step % self.sculpt_interval != 0:
            return control

        kept = sculpt_lora_params(
            model=model,
            preserve_ratio=self.preserve_ratio,
            target_keywords=self.target_keywords,
        )
        print(
            "[MITS LoRASculpt] sculpted "
            f"step={state.global_step}, tensors={len(kept)}, kept_params={sum(kept.values())}",
            flush=True,
        )
        return control


def lora_config_kwargs(rank: int = 16) -> Dict[str, object]:
    """Return PEFT LoRA config kwargs for Qwen2.5-VL single-card tuning."""
    return {
        "r": rank,
        "lora_alpha": rank * 2,
        "lora_dropout": 0.05,
        "target_modules": DEFAULT_LORA_TARGET_MODULES,
    }
