# Task-aware Targeted 改造与执行清单

## 目标

当前 `20% targeted` 的主要问题是：目标场景比例提高了，但 `counting/localization` 的 QA 级训练占比没有提高。因此本轮改造先做三件事：

1. 将 targeted 数据构建从 image/scene-level 升级为 QA-level / task-aware targeted。
2. 推理时只对 `counting/localization` 加格式约束，避免 recognition 被全局格式约束拉低。
3. 为后续 localization 训练侧改进准备低风险实验顺序。

## 1. Task-aware targeted 数据构建

### 更改内容

`build_targeted_train_set.py` 新增参数：

```bash
--task-aware
--task-quota counting=12
--task-quota localization=10
--task-quota recognition=7
--task-quota background=2
--task-quota reasoning=1
--output-suffix targeted_taskaware
```

默认不加 `--task-aware` 时保持旧行为，不影响已有 `20% targeted` 结果。

### 原理

旧 targeted 只改变选图分布，不能保证每张图的训练 messages 偏向 counting/localization。新逻辑在 ShareGPT 转换阶段按任务配额保留 QA，使训练梯度更集中到当前弱项。

### 运行命令

```bash
cd /root/autodl-tmp/MITS

python third_party/ScalSelect/mits_tools/build_targeted_train_set.py \
  --dataset-root /root/autodl-tmp/data/dataset \
  --index /root/autodl-tmp/data/outputs/full/mits_index.jsonl \
  --base-selected /root/autodl-tmp/data/outputs/full/mits_selected_15_48g_fast.jsonl \
  --output-dir /root/autodl-tmp/data/outputs/full \
  --exclude-eval /root/autodl-tmp/data/outputs/full/eval_test1000/mits_test_qas.jsonl \
  --exclude-eval /root/autodl-tmp/data/outputs/full/eval_test1000_merged/mits_test_qas.jsonl \
  --ratio 20 \
  --task-aware \
  --task-quota counting=12 \
  --task-quota localization=10 \
  --task-quota recognition=7 \
  --task-quota background=2 \
  --task-quota reasoning=1 \
  --output-suffix targeted_taskaware
```

输出：

```text
/root/autodl-tmp/data/outputs/full/mits_selected_20_targeted_taskaware.jsonl
/root/autodl-tmp/data/outputs/full/mits_selected_20_targeted_taskaware_train32_sharegpt.jsonl
```

## 2. 训练前 QA 分布检查

### 原理

不先看 QA 分布就训练，会重复旧问题：数据看似 targeted，实际 task 监督没有 targeted。

### 运行命令

```bash
python third_party/ScalSelect/mits_tools/summarize_mits_train_qa_distribution.py \
  --input /root/autodl-tmp/data/outputs/full/mits_selected_15_train32_sharegpt.jsonl \
  --label scal_15 \
  --input /root/autodl-tmp/data/outputs/full/mits_selected_20_targeted_taskaware_train32_sharegpt.jsonl \
  --label targeted_20_taskaware \
  --output-dir /root/autodl-tmp/data/outputs/full/diagnostics/train_distribution_15_vs_20target_taskaware
```

验收标准：

- `counting + localization` 占比明显高于旧 `20% targeted`。
- 如果占比没有提高，不进入训练，先调整 `--task-quota`。

## 3. 训练 20% task-aware LoRASculpt

### 原理

先只改变 QA 监督比例，保持训练配置不变，才能判断 task-aware targeted 是否有效。

### 运行命令

```bash
cd /root/autodl-tmp/MITS

export DATASET_ROOT=/root/autodl-tmp/data/dataset
export MODEL_PATH=/root/autodl-tmp/Qwen/Qwen2.5-VL-7B-Instruct
export WORK_DIR=/root/autodl-tmp/data/outputs/full
export LOG_DIR=$WORK_DIR/logs
export OMP_NUM_THREADS=8

export TRAIN_DATASET=$WORK_DIR/mits_selected_20_targeted_taskaware_train32_sharegpt.jsonl
export TRAIN_OUT=/root/autodl-tmp/data/train_outputs/mits_20_taskaware_lorasculpt_full_swanlab

export LORASCULPT_INTERVAL=300
export LORASCULPT_PRESERVE_RATIO=0.10

bash scripts/run_with_log.sh train_lorasculpt_20_taskaware \
  swift sft \
    --external_plugins scripts/swift_lorasculpt_plugin.py \
    --model "$MODEL_PATH" \
    --tuner_type lora \
    --dataset "$TRAIN_DATASET" \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --learning_rate 1e-4 \
    --max_length 4096 \
    --save_steps 300 \
    --save_total_limit 5 \
    --logging_steps 5 \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --output_dir "$TRAIN_OUT" \
    --attn_impl sdpa \
    --max_pixels 1048576 \
    --report_to tensorboard swanlab \
    --swanlab_project MITS-Qwen25VL \
    --swanlab_exp_name mits_20_taskaware_lorasculpt_train32 \
    --swanlab_mode cloud
```

预计耗时：约 `14-16 小时`。48G 卡在 `max_pixels=1048576` 下大概率够用，旧 20% targeted 同配置已验证。

## 4. Counting/Localization-only 格式约束推理

### 更改内容

`predict_qwen25vl_eval.py` 新增：

```bash
--task-format-instruction count_loc
```

`count_loc` 只对 counting/localization 追加格式约束；`auto` 保留原全局逻辑；`none` 仍为默认。

### 原理

全局 format prompt 让 counting 提升，但让 recognition 下降。`count_loc` 保留 counting/localization 的格式收益，同时避免 recognition 被额外约束。

### 运行命令

```bash
cd /root/autodl-tmp/MITS

export EVAL_BASE=/root/autodl-tmp/data/outputs/full/eval_test1000_merged
export EVAL_OUT=/root/autodl-tmp/data/outputs/full/eval_test1000_20target_taskaware_countlocprompt
mkdir -p "$EVAL_OUT/predictions" "$EVAL_OUT/scores" "$EVAL_OUT/speed"

python third_party/ScalSelect/mits_tools/predict_qwen25vl_eval.py \
  --eval-jsonl "$EVAL_BASE/mits_test_qas.jsonl" \
  --output "$EVAL_OUT/predictions/ours_20_taskaware_countlocprompt_mits_test.jsonl" \
  --model /root/autodl-tmp/Qwen/Qwen2.5-VL-7B-Instruct \
  --adapter /root/autodl-tmp/data/train_outputs/mits_20_taskaware_lorasculpt_full_swanlab/v0-YYYYMMDD-HHMMSS/checkpoint-XXXX \
  --model-label ours_20_taskaware_countlocprompt \
  --device-map cuda:0 \
  --torch-dtype bfloat16 \
  --max-pixels 2073600 \
  --max-new-tokens 128 \
  --task-format-instruction count_loc \
  --progress-every 1 \
  --resume

python third_party/ScalSelect/mits_tools/score_mits_predictions.py \
  --predictions "$EVAL_OUT/predictions/ours_20_taskaware_countlocprompt_mits_test.jsonl" \
  --output-dir "$EVAL_OUT/scores/ours_20_taskaware_countlocprompt_mits_test"
```

预计耗时：5000 QA adapter 推理约 `2-2.2 小时`。48G 推理已验证够用。

## 5. 结果对比与错误分析

```bash
python third_party/ScalSelect/mits_tools/compare_mits_eval_multi.py \
  --summary "base=$EVAL_BASE/scores/base_mits_test/summary.json" \
  --summary "traffic_full=$EVAL_BASE/scores/traffic_full_mits_test/summary.json" \
  --summary "ours_15_merged=$EVAL_BASE/scores/ours_lorasculpt_merged_mits_test/summary.json" \
  --summary "ours_20_targeted=/root/autodl-tmp/data/outputs/full/eval_test1000_20target/scores/ours_20_targeted_mits_test/summary.json" \
  --summary "ours_20_taskaware=$EVAL_OUT/scores/ours_20_taskaware_countlocprompt_mits_test/summary.json" \
  --ours-label "ours_20_taskaware" \
  --output-md "$EVAL_OUT/compare/base_traffic_ours15_ours20_taskaware_mits_test.md" \
  --output-csv "$EVAL_OUT/compare/base_traffic_ours15_ours20_taskaware_mits_test.csv"

python third_party/ScalSelect/mits_tools/analyze_mits_prediction_errors.py \
  --scores traffic_full=$EVAL_BASE/scores/traffic_full_mits_test/per_sample_scores.jsonl \
  --scores ours_15_merged=$EVAL_BASE/scores/ours_lorasculpt_merged_mits_test/per_sample_scores.jsonl \
  --scores ours_20_taskaware=$EVAL_OUT/scores/ours_20_taskaware_countlocprompt_mits_test/per_sample_scores.jsonl \
  --focus-model ours_20_taskaware \
  --reference-model traffic_full \
  --baseline-model ours_15_merged \
  --task counting \
  --task localization \
  --output-dir "$EVAL_OUT/error_analysis"
```

## 时间与显存评估

| 阶段 | 预计耗时 | 48G 是否够用 |
|---|---:|---|
| 数据构建 | 5-15 分钟 | 是 |
| QA 分布统计 | 1-5 分钟 | 是 |
| 20% task-aware 训练 | 14-16 小时 | 大概率够用 |
| 5000 QA adapter 推理 | 2-2.2 小时 | 是 |
| scoring / compare / error analysis | 1-5 分钟 | 是 |
| 可选 merged 导出 | 10-30 分钟 | 显存够，需约 16GB 磁盘 |
| 可选 merged 推理 | 1.3-1.5 小时 | 是 |

注意：`max_pixels=2073600` 的训练不保证 48G 稳定，必须先 `max_steps=20` smoke；本轮正式训练仍建议使用 `1048576`。

## 后续 localization 训练侧 ablation

只有当 `20% task-aware + count_loc prompt` 后 localization 仍明显落后 Traffic Full 时，再按以下顺序继续：

1. `20% task-aware + max_pixels 2073600`，先 smoke；
2. `20% task-aware + vanilla LoRA`，验证 LoRASculpt 是否压制 localization；
3. `20% task-aware + visual merger/projector LoRA`，验证视觉 grounding 瓶颈；
4. 最后再考虑 `30% targeted`。
