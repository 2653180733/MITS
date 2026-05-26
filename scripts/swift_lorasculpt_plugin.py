"""ms-swift plugin that injects the MITS Qwen LoRASculpt callback.

Use with:
    swift sft --external_plugins scripts/swift_lorasculpt_plugin.py ...
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from third_party.LoRASculpt.mits_qwen.qwen_lorasculpt import (  # noqa: E402
    PeriodicQwenLoRASculptCallback,
)


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


callback = PeriodicQwenLoRASculptCallback(
    sculpt_interval=_env_int("LORASCULPT_INTERVAL", 300),
    preserve_ratio=_env_float("LORASCULPT_PRESERVE_RATIO", 0.10),
)

# ms-swift external plugins are imported before trainer construction.  For
# callback plugins, the supported convention is to expose `extra_callbacks`.
extra_callbacks = [callback]

print(
    "[MITS LoRASculpt] extra_callbacks registered "
    f"(interval={callback.sculpt_interval}, preserve_ratio={callback.preserve_ratio})"
)
