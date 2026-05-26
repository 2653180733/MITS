# Qwen2.5-VL LoRASculpt Local Adapter

This folder is part of the local copy under `D:\Code\PyCode\MITS\third_party\LoRASculpt`.
The original reference repository under `D:\Desktop\Daily\Master\LLM\MITS` is not modified.

Use `qwen_lorasculpt.py` when training Qwen2.5-VL with HuggingFace Trainer, TRL, or a local LLaMA-Factory wrapper.

```python
from third_party.LoRASculpt.mits_qwen.qwen_lorasculpt import (
    PeriodicQwenLoRASculptCallback,
    lora_config_kwargs,
)

trainer.add_callback(
    PeriodicQwenLoRASculptCallback(
        sculpt_interval=300,
        preserve_ratio=0.10,
    )
)
```

Recommended single-card defaults:

| LoRA rank | Preserve ratio |
|---:|---:|
| 8 | 0.15 |
| 16 | 0.10 |
| 32 | 0.05 |
