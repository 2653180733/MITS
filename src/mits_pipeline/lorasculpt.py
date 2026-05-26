"""Small LoRASculpt helpers for PEFT LoRA modules."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

try:
    from transformers import TrainerCallback
except Exception:  # pragma: no cover - transformers exists in training env.
    TrainerCallback = object


DEFAULT_SCULPT_TARGET_KEYWORDS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "mm_projector",
    "multi_modal_projector",
    "merger",
]

DEFAULT_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def recommended_preserve_ratio(lora_rank: int) -> float:
    """Return the default preserve ratio for rank-sculpt co-design."""
    if lora_rank <= 8:
        return 0.15
    if lora_rank <= 16:
        return 0.10
    return 0.05


def is_sculpt_target(name: str, target_keywords: Optional[Iterable[str]] = None) -> bool:
    keywords = list(target_keywords or DEFAULT_SCULPT_TARGET_KEYWORDS)
    return "lora_" in name and any(keyword in name for keyword in keywords)


def sculpt_lora_params(
    model,
    preserve_ratio: float = 0.10,
    target_keywords: Optional[Iterable[str]] = None,
) -> Dict[str, int]:
    """Zero low-magnitude LoRA parameters in target modules.

    This function is intentionally framework-light: call it from a Trainer
    callback every ``sculpt_interval`` optimizer steps.
    """
    if preserve_ratio <= 0 or preserve_ratio > 1:
        raise ValueError("preserve_ratio must be in (0, 1].")

    stats: Dict[str, int] = {}
    for name, param in model.named_parameters():
        if not is_sculpt_target(name, target_keywords=target_keywords):
            continue
        if not getattr(param, "requires_grad", False):
            continue

        data = param.data
        flat_abs = data.abs().flatten()
        keep = max(1, int(flat_abs.numel() * preserve_ratio))
        threshold = flat_abs.topk(keep).values.min()
        mask = data.abs() >= threshold
        data.mul_(mask)
        stats[name] = int(mask.sum().item())

    return stats


class PeriodicLoRASculptCallback(TrainerCallback):
    """Minimal HuggingFace Trainer callback for periodic LoRA sculpting."""

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
        if state.global_step <= 0 or state.global_step % self.sculpt_interval != 0:
            return control
        sculpt_lora_params(
            model=model,
            preserve_ratio=self.preserve_ratio,
            target_keywords=self.target_keywords,
        )
        return control
