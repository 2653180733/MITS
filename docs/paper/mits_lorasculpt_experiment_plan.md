# MITS 15% LoRASculpt 实验与论文思路整理

## 当前状态

已经完成的核心工作：

- 构建了 MITS 15% 训练集：`mits_selected_15_train32_sharegpt.jsonl`。
- 使用 Qwen2.5-VL-7B-Instruct 训练了 `Ours 15% + LoRASculpt`。
- 搭建了 Base Qwen、MITS Traffic Full、Ours 三模型评估流水线。
- 在 `eval_test1000` 上完成 1000 张图片、5000 条 QA 的 internal holdout 评估。

当前 internal holdout 自动评分结果：

| 指标 | Base | Traffic Full | Ours 15% | Ours - Base | Ours - Traffic |
|---|---:|---:|---:|---:|---:|
| Recognition | 82.90 | 98.70 | 95.30 | +12.40 | -3.40 |
| Counting | 80.21 | 95.00 | 85.00 | +4.79 | -10.00 |
| Localization | 59.82 | 85.77 | 79.47 | +19.65 | -6.30 |
| Automated Avg | 74.31 | 93.15 | 86.59 | +12.28 | -6.57 |

结论应表述为：Ours 使用 15% 数据显著超过 Base Qwen，并达到 Traffic Full 自动平均分的约 92.96%。Traffic Full 是全量 MITS 训练模型，在 internal holdout 上应作为 full-data upper bound，而不是同预算公平 baseline。

## 论文主线

建议论文问题定义：

> 通用 VLM 在交通监控场景中的识别、计数和定位能力不足；全量微调成本高。本文研究如何用少量高价值交通数据进行高效适配。

建议贡献点：

- 数据效率：用 15% 高价值 MITS 数据显著提升交通 QA 能力。
- LoRASculpt：以参数高效方式适配 Qwen2.5-VL。
- 评估协议：同时报告 internal holdout、official test 和 external generalization。
- 部署效率：导出 merged Ours，比较 adapter 与 merged 部署形态的推理速度。

## 下一步实验

### 1. Official / External Eval

优先跑作者 official test：

```bash
cd /root/autodl-tmp/MITS

export EVAL_KIND=official
export AUTHOR_TEST_PATH=/path/to/swift_data/v1.0_test.jsonl
export EVAL_DIR=/root/autodl-tmp/data/outputs/full/eval_official
export MAX_PIXELS=2073600
unset EVAL_LIMIT

bash scripts/run_mits_eval_base_traffic_ours.sh
```

如果没有 official test，跑外部交通 QA：

```bash
python third_party/ScalSelect/mits_tools/convert_external_vqa_to_eval.py \
  --input /path/to/external.jsonl \
  --output /root/autodl-tmp/data/outputs/full/eval_external/external_eval.jsonl \
  --scene external \
  --source-name nuscenes_qa

export EVAL_KIND=external
export EVAL_JSONL=/root/autodl-tmp/data/outputs/full/eval_external/external_eval.jsonl
export EVAL_DIR=/root/autodl-tmp/data/outputs/full/eval_external
bash scripts/run_mits_eval_base_traffic_ours.sh
```

### 2. Merged Ours 速度和精度

先导出 merged Ours：

```bash
python third_party/ScalSelect/mits_tools/export_merged_lora_qwen25vl.py \
  --base-model /root/autodl-tmp/Qwen/Qwen2.5-VL-7B-Instruct \
  --adapter /root/autodl-tmp/data/train_outputs/mits_15_lorasculpt_full_swanlab/v0-20260523-162150/checkpoint-1540 \
  --output /root/autodl-tmp/data/train_outputs/mits_15_lorasculpt_merged
```

四模型 smoke：

```bash
export EVAL_DIR=/root/autodl-tmp/data/outputs/full/eval_speed_merged
export EVAL_LIMIT=200
export MAX_PIXELS=1048576
bash scripts/run_mits_eval_base_traffic_ours_merged.sh
```

正式四模型评估：

```bash
export EVAL_DIR=/root/autodl-tmp/data/outputs/full/eval_test1000_merged
unset EVAL_LIMIT
export MAX_PIXELS=2073600
bash scripts/run_mits_eval_base_traffic_ours_merged.sh
```

速度表应包含 Base、Traffic Full、Ours adapter、Ours merged 的 `avg_s_per_qa`、P50/P90/P95、平均生成 token 数。

### 3. Targeted 20%-30% 训练

当前 Ours 最大短板集中在：

- `accident / construction / firesmoke`
- `counting / localization`

构建 targeted 数据：

```bash
python third_party/ScalSelect/mits_tools/build_targeted_train_set.py \
  --dataset-root /root/autodl-tmp/data/dataset \
  --index /root/autodl-tmp/data/outputs/full/mits_index.jsonl \
  --base-selected /root/autodl-tmp/data/outputs/full/mits_selected_15.jsonl \
  --output-dir /root/autodl-tmp/data/outputs/full \
  --exclude-eval /root/autodl-tmp/data/outputs/full/eval_test1000/mits_test_qas.jsonl \
  --ratio 20 \
  --ratio 30
```

训练顺序：

| 实验 | 数据 | LoRA | 目的 |
|---|---:|---|---|
| Ours-15 | 15% | r16 | 已完成 |
| Ours-20-target | 20% | r16 | 验证 targeted 数据是否有效 |
| Ours-30-target | 30% | r16 | 缩小 Traffic Full 差距 |
| Ours-30-r32 | 30% | r32 | 验证容量是否是瓶颈 |

## 论文表格建议

建议最终至少包含：

- Table 1：Internal MITS holdout，重点展示 Ours vs Base，Traffic Full 标注为 full-data upper bound。
- Table 2：Official MITS test，作为与作者模型比较的主表。
- Table 3：External traffic QA，展示泛化性能。
- Table 4：Speed / deployment，比较 Base、Traffic Full、Ours adapter、Ours merged。
- Table 5：Ablation，比较 15%、20%-target、30%-target、30%-r32。

## 风险与注意事项

- 不覆盖 `eval_test1000`，该目录作为冻结基线保留。
- 所有新实验必须使用新 `EVAL_DIR`。
- Traffic Full 在 internal MITS holdout 上可能有训练数据优势，不能把它作为完全公平同预算 baseline。
- `background` 和 `reasoning` 需要 judge 分数后才能报告完整 MITS Avg。
- 如果 merged Ours 精度和 adapter Ours 不一致，需要检查 LoRA merge 是否成功、processor 是否保存完整、推理参数是否完全一致。
