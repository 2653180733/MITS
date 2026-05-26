# MITS 本地改造版流水线

日期：2026-05-11

## 1. 代码组织

原始参考代码保留不动：

```text
D:\Desktop\Daily\Master\LLM\MITS
```

当前项目内的可修改复制件：

```text
D:\Code\PyCode\MITS\third_party\ScalSelect
D:\Code\PyCode\MITS\third_party\LoRASculpt
D:\Code\PyCode\MITS\third_party\UnicomBenchmark
```

本项目新增的共享工具：

```text
D:\Code\PyCode\MITS\src\mits_pipeline
```

MITS 数据集默认路径：

```text
D:\Code\PyCode\multi_traffic\src\data\archives\dataset
```

所有后续代码修改都应发生在 `D:\Code\PyCode\MITS` 下，不要改原始参考目录。

## 2. 本地复制件改动说明

`third_party\ScalSelect`：

- 新增 `mits_tools\build_mits_index.py`：构建 MITS JSONL 索引。
- 新增 `mits_tools\convert_mits_to_sharegpt.py`：转换 MITS 为 ShareGPT。
- 新增 `mits_tools\select_mits_subset.py`：融合 CUR、任务覆盖、长尾权重选择子集。
- 新增 `mits_tools\run_mits_pipeline.ps1`：本地一键流水线。
- 修改 `scripts\feature_extract_sft.py`：支持 Windows 路径和 `--image-base-path`，避免硬编码源码里的 base path。

`third_party\LoRASculpt`：

- 新增 `mits_qwen\qwen_lorasculpt.py`：Qwen2.5-VL/PEFT/Trainer 可复用的 LoRASculpt callback。
- 新增 `mits_qwen\README.md`：接入说明。

`third_party\UnicomBenchmark`：

- 当前仅复制保留，作为评估基准参考，不做源码改动。

## 3. 安全预览流程

默认只读取前 100 条样本，确认格式和路径无误：

```powershell
$env:PYTHONPATH = "D:\Code\PyCode\MITS\src;" + $env:PYTHONPATH

python third_party\ScalSelect\mits_tools\build_mits_index.py `
  --dataset-root "D:\Code\PyCode\multi_traffic\src\data\archives\dataset" `
  --output "outputs\preview\mits_index.jsonl" `
  --limit 100

python third_party\ScalSelect\mits_tools\convert_mits_to_sharegpt.py `
  --dataset-root "D:\Code\PyCode\multi_traffic\src\data\archives\dataset" `
  --index "outputs\preview\mits_index.jsonl" `
  --output "outputs\preview\mits_sharegpt.json" `
  --max-pairs-per-sample 32
```

也可以用一键脚本跳过特征提取，只验证数据转换和子集选择：

```powershell
third_party\ScalSelect\mits_tools\run_mits_pipeline.ps1 `
  -DatasetRoot "D:\Code\PyCode\multi_traffic\src\data\archives\dataset" `
  -ModelPath "D:\models\Qwen2.5-VL-7B-Instruct" `
  -WorkDir "D:\Code\PyCode\MITS\outputs\preview" `
  -Limit 100 `
  -Ratio 15 `
  -SkipFeatureExtraction
```

## 4. 正式数据准备

全量扫描必须显式加 `--allow-full-scan`，避免误扫 17W 图像对应的海量 JSON：

```powershell
$env:PYTHONPATH = "D:\Code\PyCode\MITS\src;" + $env:PYTHONPATH

python third_party\ScalSelect\mits_tools\build_mits_index.py `
  --dataset-root "D:\Code\PyCode\multi_traffic\src\data\archives\dataset" `
  --output "outputs\full\mits_index.jsonl" `
  --limit 0 `
  --allow-full-scan

python third_party\ScalSelect\mits_tools\convert_mits_to_sharegpt.py `
  --dataset-root "D:\Code\PyCode\multi_traffic\src\data\archives\dataset" `
  --index "outputs\full\mits_index.jsonl" `
  --output "outputs\full\mits_sharegpt.json" `
  --max-pairs-per-sample 32
```

生成的 ShareGPT 样本字段：

- `id`：本地连续数字 ID，用于和 CUR `sample_id` 对齐。
- `messages`：多轮 user/assistant 对话。
- `images`：绝对图像路径。
- `meta`：原始 MITS ID、JSON 路径、任务标签、长尾标签。

## 5. ScalSelect 特征提取和 CUR

使用当前项目内的 ScalSelect 复制件：

```powershell
python third_party\ScalSelect\scripts\feature_extract_sft.py `
  --model "D:\models\Qwen2.5-VL-7B-Instruct" `
  --model-type qwen `
  --dataset "D:\Code\PyCode\MITS\outputs\full\mits_sharegpt.json" `
  --output-dir "D:\Code\PyCode\MITS\outputs\full\features" `
  --max-samples -1 `
  --sample-batch-size 1 `
  --torch-dtype bfloat16 `
  --max-length 4096
```

如果 ShareGPT 中保存的是相对图像路径，可额外传：

```powershell
--image-base-path "D:\Code\PyCode\multi_traffic\src\data\archives\dataset\images"
```

计算 CUR 分数：

```powershell
python third_party\ScalSelect\scripts\cur.py `
  --features-dir "D:\Code\PyCode\MITS\outputs\full\features" `
  --output "D:\Code\PyCode\MITS\outputs\full\importance_scores.jsonl" `
  --sv-threshold 0.9
```

## 6. 任务感知子集选择

选择 15% 子集：

```powershell
python third_party\ScalSelect\mits_tools\select_mits_subset.py `
  --index "outputs\full\mits_index.jsonl" `
  --cur-scores "outputs\full\importance_scores.jsonl" `
  --output "outputs\full\mits_selected_15.jsonl" `
  --ratio 15 `
  --lambda-task 0.2 `
  --lambda-rare 0.1

python third_party\ScalSelect\mits_tools\convert_mits_to_sharegpt.py `
  --dataset-root "D:\Code\PyCode\multi_traffic\src\data\archives\dataset" `
  --index "outputs\full\mits_selected_15.jsonl" `
  --output "outputs\full\mits_selected_15_sharegpt.json" `
  --max-pairs-per-sample 32
```

建议生成以下比例用于论文曲线：

```text
5%, 10%, 15%, 20%, 30%
```

核心指标：

- MITS Average 与 Full-LoRA gap <= 3%。
- MMBench/MME/SEED-Bench 平均保持率 >= 90%。

## 7. QLoRA + LoRASculpt 接入

默认配置：

```text
QLoRA int4
bf16
gradient_checkpointing = true
LoRA rank = 16
LoRA alpha = 32
LoRA dropout = 0.05
sculpt_interval = 300
preserve_ratio = 0.10
```

本地 LoRASculpt 复制件接入：

```python
from third_party.LoRASculpt.mits_qwen.qwen_lorasculpt import (
    PeriodicQwenLoRASculptCallback,
    lora_config_kwargs,
)

lora_kwargs = lora_config_kwargs(rank=16)

trainer.add_callback(
    PeriodicQwenLoRASculptCallback(
        sculpt_interval=300,
        preserve_ratio=0.10,
    )
)
```

rank 与 preserve ratio 默认关系：

| LoRA rank | Preserve ratio |
|---:|---:|
| 8 | 0.15 |
| 16 | 0.10 |
| 32 | 0.05 |

如果使用 TRL 或 LLaMA-Factory，不修改其安装包源码；在当前项目下写 wrapper，将 `D:\Code\PyCode\MITS\src` 和项目根目录加入 `PYTHONPATH`，再导入 callback。

## 8. 一键脚本示例

全量流水线示例：

```powershell
third_party\ScalSelect\mits_tools\run_mits_pipeline.ps1 `
  -DatasetRoot "D:\Code\PyCode\multi_traffic\src\data\archives\dataset" `
  -ModelPath "D:\models\Qwen2.5-VL-7B-Instruct" `
  -WorkDir "D:\Code\PyCode\MITS\outputs\full" `
  -Limit 0 `
  -AllowFullScan `
  -Ratio 15 `
  -MaxPairsPerSample 32
```

说明：

- `-Limit 0 -AllowFullScan` 表示确认进行全量流式扫描。
- 特征提取会加载 Qwen2.5-VL，需按单卡显存调整 batch、长度和量化配置。
- 如只做格式检查，加 `-SkipFeatureExtraction`。

## 9. MITS 评估

MITS 论文五类能力：

| 指标 | 含义 | 计算方式 |
|---|---|---|
| `S_recog` | 目标/事件识别 | Yes/No 或类别准确率 |
| `S_count` | 目标计数 | `max(0, 1 - abs(pred-gt)/max(gt,1))` |
| `S_loc` | 目标定位 | 预测框与 GT 框 IoU |
| `S_background` | 背景分析 | DeepSeek-R1 等评分模型 0-1 打分 |
| `S_reasoning` | 事件推理 | 事实一致性与语义一致性 0-1 打分 |

总分：

```text
MITS Average = mean(S_background, S_recog, S_count, S_loc, S_reasoning)
```

通用能力保持率：

```text
Retention = fine_tuned_score / original_qwen_score * 100%
```

最终论文表格至少包含：

- Base Qwen2.5-VL。
- Full-LoRA。
- Random 5/10/15/20/30%。
- ScalSelect 5/10/15/20/30%。
- Task-aware + rarity-balanced + LoRASculpt 5/10/15/20/30%。
