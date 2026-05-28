# Targeted 20% 效果偏弱的排查与改进计划

当前现象：`20% targeted` 相比 `15% LoRASculpt merged` 的 `Automated Avg` 只提升约 `+1.01`，其中 `localization` 只提升约 `+0.25`。这说明 targeted 方向可能有效，但信号偏弱，需要先定位瓶颈，再决定是否重训。

## 1. 统计训练集 QA 级分布

成本最低，不需要 GPU。目的：确认 targeted 20% 新增数据是否真的增加了 `counting/localization` QA，而不是只增加了目标场景图片。

```bash
cd /root/autodl-tmp/MITS

python third_party/ScalSelect/mits_tools/summarize_mits_train_qa_distribution.py \
  --input /root/autodl-tmp/data/outputs/full/mits_selected_15_train32_sharegpt.jsonl \
  --label scal_15 \
  --input /root/autodl-tmp/data/outputs/full/mits_selected_20_targeted_train32_sharegpt.jsonl \
  --label targeted_20 \
  --output-dir /root/autodl-tmp/data/outputs/full/diagnostics/train_distribution_15_vs_20target
```

重点看：

- `train_qa_task_distribution.csv`
- `train_scene_distribution.csv`
- `train_qa_distribution.md`

如果 `targeted_20` 的 `counting/localization` QA 占比没有明显增加，问题在数据构建粒度。

## 2. 做模型间错误分析

成本低，不需要 GPU。目的：找出 `Ours 20%` 错但 `Traffic Full` 对的样本，并比较 `Ours 20%` 相比 `Ours 15%` 的改进和退化。

```bash
cd /root/autodl-tmp/MITS

export EVAL15=/root/autodl-tmp/data/outputs/full/eval_test1000_merged
export EVAL20=/root/autodl-tmp/data/outputs/full/eval_test1000_20target

python third_party/ScalSelect/mits_tools/analyze_mits_prediction_errors.py \
  --scores base=$EVAL15/scores/base_mits_test/per_sample_scores.jsonl \
  --scores traffic_full=$EVAL15/scores/traffic_full_mits_test/per_sample_scores.jsonl \
  --scores ours_15_merged=$EVAL15/scores/ours_lorasculpt_merged_mits_test/per_sample_scores.jsonl \
  --scores ours_20_targeted=$EVAL20/scores/ours_20_targeted_mits_test/per_sample_scores.jsonl \
  --focus-model ours_20_targeted \
  --reference-model traffic_full \
  --baseline-model ours_15_merged \
  --task counting \
  --task localization \
  --output-dir /root/autodl-tmp/data/outputs/full/diagnostics/error_analysis_20target
```

重点看：

- `score_by_task.csv`
- `score_by_scene.csv`
- `focus_wrong_reference_right.csv`
- `focus_regressions_vs_baseline.csv`
- `focus_improvements_vs_baseline.csv`

如果 `focus_wrong_reference_right.csv` 里大量是 bbox 格式不规范，则先改推理 prompt；如果是图像细节看不清，则考虑训练分辨率或视觉侧 LoRA。

## 3. 尝试格式约束推理

成本中等，只需要重新推理，不需要重训。目的：排除 `counting/localization` 因输出格式不符合 scorer 而被低估。

新增参数：

```bash
--task-format-instruction auto
```

示例：

```bash
cd /root/autodl-tmp/MITS

export EVAL_BASE=/root/autodl-tmp/data/outputs/full/eval_test1000_merged
export EVAL_OUT=/root/autodl-tmp/data/outputs/full/eval_test1000_20target_formatprompt
mkdir -p "$EVAL_OUT/predictions" "$EVAL_OUT/scores"

python third_party/ScalSelect/mits_tools/predict_qwen25vl_eval.py \
  --eval-jsonl "$EVAL_BASE/mits_test_qas.jsonl" \
  --output "$EVAL_OUT/predictions/ours_20_targeted_formatprompt_mits_test.jsonl" \
  --model /root/autodl-tmp/Qwen/Qwen2.5-VL-7B-Instruct \
  --adapter /root/autodl-tmp/data/train_outputs/mits_20_targeted_lorasculpt_full_swanlab/v0-20260527-232648/checkpoint-2053 \
  --model-label ours_20_targeted_formatprompt \
  --device-map cuda:0 \
  --torch-dtype bfloat16 \
  --max-pixels 2073600 \
  --max-new-tokens 128 \
  --task-format-instruction auto \
  --progress-every 1 \
  --resume

python third_party/ScalSelect/mits_tools/score_mits_predictions.py \
  --predictions "$EVAL_OUT/predictions/ours_20_targeted_formatprompt_mits_test.jsonl" \
  --output-dir "$EVAL_OUT/scores/ours_20_targeted_formatprompt_mits_test"
```

如果 counting/localization 明显提升，说明问题不完全是训练，而是输出格式约束不足。

## 4. 补 20% non-targeted ScalSelect 对照

成本高，需要重新训练。目的：证明 targeted 的提升不是单纯来自 `15% -> 20%` 的数据量增加。

先构建同样排除评估集的 20% 非 targeted 数据：

```bash
cd /root/autodl-tmp/MITS

python third_party/ScalSelect/mits_tools/build_nontargeted_scalselect_train_set.py \
  --dataset-root /root/autodl-tmp/data/dataset \
  --selected-index /root/autodl-tmp/data/outputs/full/mits_selected_20_48g_fast.jsonl \
  --output-index /root/autodl-tmp/data/outputs/full/mits_selected_20_scalselect_evalsafe.jsonl \
  --output-sharegpt /root/autodl-tmp/data/outputs/full/mits_selected_20_scalselect_evalsafe_train32_sharegpt.jsonl \
  --exclude-eval /root/autodl-tmp/data/outputs/full/eval_test1000/mits_test_qas.jsonl \
  --exclude-eval /root/autodl-tmp/data/outputs/full/eval_test1000_merged/mits_test_qas.jsonl
```

然后使用和 targeted 20% 相同的训练配置，只改：

```bash
export TRAIN_DATASET=/root/autodl-tmp/data/outputs/full/mits_selected_20_scalselect_evalsafe_train32_sharegpt.jsonl
export TRAIN_OUT=/root/autodl-tmp/data/train_outputs/mits_20_scalselect_lorasculpt_full_swanlab
export SWANLAB_EXP_NAME=mits_20_scalselect_lorasculpt_train32
```

比较：

- `15% LoRASculpt`
- `20% ScalSelect LoRASculpt`
- `20% Targeted LoRASculpt`

只有 `20% Targeted > 20% ScalSelect`，targeted 贡献才站得住。

## 5. 训练侧改进实验

成本最高，最后再做。重点不是继续盲目加到 30%，而是验证瓶颈是不是视觉 grounding。

优先实验顺序：

1. `20% targeted + vanilla LoRA`：验证 LoRASculpt 是否压制 counting/localization。
2. `20% targeted + max_pixels 2073600`：训练和评估分辨率对齐，验证小目标/定位是否受益。
3. `20% targeted + aligner/projector/visual merger LoRA`：验证冻结视觉侧是否限制 localization。
4. `30% targeted`：只有当前三项排查完成后再做。

论文里最有价值的结论不是“数据越多越好”，而是说明：

- 选择哪些数据有效；
- 选择策略相比同数据量 baseline 是否有效；
- 推理格式、训练分辨率、视觉侧可训练参数如何影响 counting/localization。
