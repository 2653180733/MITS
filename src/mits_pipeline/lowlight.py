"""Low-light image detection and enhancement utilities for MITS."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


DEFAULT_LOWLIGHT_THRESHOLD = 80.0
LOWLIGHT_META_KEYWORDS = ("night", "low", "dark", "dim")


def _walk_values(value: Any) -> Iterable[Any]:
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_values(child)
    else:
        yield value


def load_rgb_image(path: str) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def compute_mean_luminance(image: Image.Image) -> float:
    arr = np.asarray(image).astype(np.float32)
    luminance = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    return float(luminance.mean())


def is_low_light_by_luminance(
    image: Image.Image,
    threshold: float = DEFAULT_LOWLIGHT_THRESHOLD,
) -> Tuple[bool, float]:
    mean_luminance = compute_mean_luminance(image)
    return mean_luminance < threshold, mean_luminance


def is_low_light_by_meta(
    item: Dict[str, Any],
    json_path: Optional[str] = None,
    rare_tags: Optional[Iterable[str]] = None,
) -> bool:
    if rare_tags and "low_light" in set(rare_tags):
        return True

    for field in ("baselabel", "categorylabel", "scene", "image"):
        for value in _walk_values(item.get(field)):
            if isinstance(value, str):
                lowered = value.lower()
                if any(keyword in lowered for keyword in LOWLIGHT_META_KEYWORDS):
                    return True

    if json_path:
        lowered_path = str(json_path).lower().replace("\\", "/")
        if any(keyword in lowered_path for keyword in LOWLIGHT_META_KEYWORDS):
            return True

    return False


def _apply_gamma(image: Image.Image, gamma: float) -> Image.Image:
    if gamma <= 0:
        raise ValueError("gamma must be positive.")
    arr = np.asarray(image).astype(np.float32) / 255.0
    corrected = np.power(arr, gamma)
    corrected = np.clip(corrected * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(corrected, mode="RGB")


def _protect_highlights(original: Image.Image, enhanced: Image.Image) -> Image.Image:
    original_arr = np.asarray(original).astype(np.float32)
    enhanced_arr = np.asarray(enhanced).astype(np.float32)
    luminance = (
        0.299 * original_arr[..., 0]
        + 0.587 * original_arr[..., 1]
        + 0.114 * original_arr[..., 2]
    )
    highlight_mask = np.clip((luminance - 190.0) / 65.0, 0.0, 1.0)[..., None]
    protected = enhanced_arr * (1.0 - 0.55 * highlight_mask) + original_arr * (
        0.55 * highlight_mask
    )
    return Image.fromarray(np.clip(protected, 0, 255).astype(np.uint8), mode="RGB")


def enhance_lowlight_image(
    image: Image.Image,
    gamma: float = 0.6,
    contrast: float = 1.25,
    sharpness: float = 1.1,
) -> Image.Image:
    enhanced = _apply_gamma(image, gamma)
    enhanced = ImageEnhance.Contrast(enhanced).enhance(contrast)
    enhanced = enhanced.filter(ImageFilter.GaussianBlur(radius=0.35))
    enhanced = ImageEnhance.Sharpness(enhanced).enhance(sharpness)
    return _protect_highlights(image, enhanced)


def enhanced_relative_path(image_rel: str) -> str:
    source = Path(str(image_rel).replace("\\", "/"))
    suffix = source.suffix or ".jpg"
    name = f"{source.stem}_lowlight_aug{suffix}"
    parent = source.parent
    return str(parent / name).replace("\\", "/") if str(parent) != "." else name


def save_lowlight_enhanced_image(
    source_path: str,
    image_root: str,
    output_root: str,
    image_rel: str,
    threshold: float = DEFAULT_LOWLIGHT_THRESHOLD,
    gamma: float = 0.6,
    contrast: float = 1.25,
    sharpness: float = 1.1,
) -> Dict[str, Any]:
    original = load_rgb_image(source_path)
    lowlight_by_luminance, mean_luminance = is_low_light_by_luminance(
        original,
        threshold=threshold,
    )
    enhanced = enhance_lowlight_image(
        original,
        gamma=gamma,
        contrast=contrast,
        sharpness=sharpness,
    )

    rel_path = enhanced_relative_path(image_rel)
    output_path = os.path.normpath(os.path.join(output_root, rel_path))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    enhanced.save(output_path, quality=95)

    return {
        "image_path": output_path,
        "image_rel": os.path.relpath(output_path, image_root),
        "mean_luminance": mean_luminance,
        "lowlight_by_luminance": lowlight_by_luminance,
    }
