# Targeted 数据与 Localization 改进计划

## 当前问题总结

当前 `20% targeted` 实验说明 targeted 方向有一定收益，但收益偏弱。与 `15% LoRASculpt merged` 相比，`20% targeted` 的 `Automated Avg` 只提升约 `+1.01`，主要表现为：

| 指标 | 15% LoRASculpt Merged | 20% Targeted | 变化 |
|---|---:|---:|---:|
| Recognition | 95.00 | 96.20 | +1.20 |
| Counting | 84.71 | 86.30 | +1.59 |
| Localization | 79.54 | 79.79 | +0.25 |
| Automated Avg | 86.42 | 87.43 | +1.01 |

训练集分布分析显示，当前 targeted 构建确实改变了场景分布：`accident / construction / firesmoke` 三类目标场景的样本占比从约 `38.54%` 提升到约 `51.43%`。这说明 scene-level targeted 是生效的。

但 QA 任务分布几乎没有改变：

| 任务 | 15% ScalSelect | 20% Targeted | 变化 |
|---|---:|---:|---:|
| Recognition | 42.37% | 42.85% | +0.48 |
| Counting | 24.20% | 24.16% | -0.04 |
| Localization | 18.91% | 18.60% | -0.31 |
| Reasoning | 9.87% | 9.84% | -0.03 |
| Background | 4.65% | 4.55% | -0.10 |

因此，当前问题不是“目标场景没选中”，而是“没有真正提高 counting/localization QA 的训练权重”。换句话说，现在的 targeted 更像是增加了困难交通场景图片，但没有让模型在训练时更频繁地学习计数和定位任务。

这也解释了为什么 localization 几乎没有提升：它需要更强的视觉 grounding 能力，而当前新增数据没有显著增加 localization 监督信号。

## 1. 改 targeted 数据构建方式

### 问题来源

当前 targeted 数据构建主要发生在 image/scene 层面：优先选择 `accident`、`construction`、`firesmoke` 等关键交通场景，再转换成 ShareGPT 训练样本。

这种方式能增加目标场景比例，但不能保证每张图中被保留的 QA 都偏向 `counting` 和 `localization`。由于 MITS 每张图通常包含多类 QA，转换成 `train32` 后，messages 中仍然保留了大量 recognition、background、reasoning 问答。结果是：

- 目标场景增加了；
- 目标任务占比没有增加；
- recognition 仍然是最大任务来源；
- counting/localization 的训练信号被大量非目标任务稀释。

这就是 `20% targeted` 只小幅提升的主要原因。

### 原理

targeted 数据应该从 scene-aware targeted 升级为 QA-level / task-aware targeted。核心思想是：

> 不只是选择“哪些图片进入训练”，还要控制“每张图片中哪些 QA 更频繁进入训练”。

对于当前短板，训练数据构建应同时满足两个目标：

1. 场景上继续强调 `accident / construction / firesmoke`；
2. 任务上强制提高 `counting / localization` 的 QA 占比。

这样模型接收到的梯度才会真正偏向当前弱项，而不是继续主要优化 recognition。

### 具体做法

保留现有 targeted 选图策略，但在 ShareGPT 转换阶段增加任务配额控制：

- 每张图优先保留 `counting` QA；
- 其次保留 `localization` QA；
- 再少量保留 `recognition` QA，避免通用识别能力退化；
- `background` 和 `reasoning` 只保留少量代表性样本；
- 对目标场景中的 counting/localization QA 给予更高采样权重。

建议目标分布为：

| 任务 | 当前 20% Targeted | 建议目标 |
|---|---:|---:|
| Counting | 24.16% | 30%-35% |
| Localization | 18.60% | 25%-30% |
| Recognition | 42.85% | 25%-30% |
| Background + Reasoning | 14.39% | 10%-15% |

这个目标不是为了完全牺牲 recognition，而是让训练监督更匹配当前模型短板。

### 预期收益

如果 QA-level targeted 生效，预期应该看到：

- counting 提升幅度大于当前 `+1.59`；
- localization 不再只提升 `+0.25`；
- recognition 可能小幅波动，但不应大幅下降；
- `20% QA-level targeted` 应明显优于 `20% non-targeted ScalSelect`。

### 验证方式

训练前必须先重新统计 QA 级分布：

```bash
python third_party/ScalSelect/mits_tools/summarize_mits_train_qa_distribution.py \
  --input /root/autodl-tmp/data/outputs/full/mits_selected_15_train32_sharegpt.jsonl \
  --label scal_15 \
  --input /root/autodl-tmp/data/outputs/full/mits_selected_20_targeted_taskaware_train32_sharegpt.jsonl \
  --label targeted_20_taskaware \
  --output-dir /root/autodl-tmp/data/outputs/full/diagnostics/train_distribution_15_vs_20target_taskaware
```

只有当 `counting/localization` 占比确实提高后，才值得开始训练。

## 2. 推理时只对 counting/localization 加格式约束

### 问题来源

格式约束推理实验显示：

| 指标 | 20% Targeted 原始推理 | 20% Targeted Format Prompt | 变化 |
|---|---:|---:|---:|
| Recognition | 96.20 | 95.20 | -1.00 |
| Counting | 86.30 | 88.20 | +1.90 |
| Localization | 79.79 | 80.13 | +0.34 |
| Automated Avg | 87.43 | 87.84 | +0.41 |

这说明 counting 的一部分错误不是模型完全不会，而是输出形式不利于自动 scorer 解析。例如模型可能回答完整自然语言，而 scorer 只需要一个明确数字。

但 recognition 加格式约束后下降了 `-1.00`，说明不能对所有任务统一加强输出限制。

### 原理

自动评分器对不同任务的输出敏感度不同：

- counting 需要解析整数；
- localization 需要解析 bbox 或 no-object；
- recognition 通常是 yes/no 或短文本，但过强约束可能改变模型原有判断方式；
- background/reasoning 是开放式问题，本来就需要 judge 或人工评分。

因此，格式约束应该按任务选择性启用，而不是全局启用。

### 具体做法

推理时只对 `counting` 和 `localization` 追加格式约束：

```text
counting:
Answer with a single integer only.

localization:
Answer only with bounding box coordinates in [x1, y1, x2, y2] format. If the target is absent, answer: no object.
```

对于其他任务：

```text
recognition / background / reasoning:
保持原始 prompt，不追加格式约束。
```

这样可以保留 counting 的格式收益，同时避免 recognition 因过度约束而下降。

### 预期收益

基于已有实验，选择性格式约束的预期收益是：

- counting 至少保留大部分 `+1.90` 的提升；
- recognition 不再出现 `-1.00` 的下降；
- localization 可能小幅提升，但不应期待大幅改善；
- `Automated Avg` 有望高于当前全局 format prompt 的 `87.84`。

### 验证方式

使用同一份 `mits_test_qas.jsonl`，只改变推理 prompt 策略，比较：

- `20% targeted` 原始推理；
- `20% targeted` 全局 format prompt；
- `20% targeted` counting/localization-only format prompt。

验收标准：

- counting 高于原始推理；
- recognition 不低于原始推理或最多轻微下降；
- localization 不低于原始推理；
- 总体 automated avg 高于 `87.43`。

## 3. Localization 的训练侧改进

### 问题来源

格式约束对 localization 的提升很小，只从 `79.79` 提升到 `80.13`，变化约 `+0.34`。这说明 localization 的主要问题不是输出格式，而是模型本身的视觉定位能力不足。

当前训练配置中，LoRA 主要作用于 language model，训练日志显示视觉侧和对齐侧基本冻结：

```text
freeze_vit=True
freeze_aligner=True
```

这种设置对 recognition 类任务相对友好，因为模型更多是在已有视觉特征上学习交通语义表达；但对 localization 和细粒度 counting 来说，模型需要更强的空间理解、目标分离和视觉 grounding 能力。仅调整语言层参数，提升空间定位能力的上限有限。

### 原理

localization 的核心不是“如何表达答案”，而是“模型是否能在图像中准确定位目标”。它依赖：

- 图像分辨率是否足够保留小目标；
- 视觉 encoder 或视觉连接层是否能适配交通场景；
- 视觉 token 到语言 token 的映射是否能表达空间信息；
- bbox 格式训练样本是否足够密集。

如果视觉侧完全冻结，模型只能在已有视觉表征上学习回答方式，难以显著改善空间 grounding。

### 具体做法

建议按成本从低到高做训练侧实验。

第一，训练和评估分辨率对齐：

```text
当前训练 max_pixels = 1048576
当前评估 max_pixels = 2073600
建议实验 max_pixels = 2073600
```

原因是 localization 和 counting 常受小目标影响。训练时图像 token 较少，可能导致模型没有学习到高分辨率下的细粒度视觉线索。

第二，做 `20% targeted + vanilla LoRA` 对照：

```text
20% targeted + LoRASculpt
vs
20% targeted + vanilla LoRA
```

目的是确认 LoRASculpt 的参数裁剪/保留机制是否对 localization 有副作用。如果 vanilla LoRA 的 localization 更高，说明 LoRASculpt 需要调整 preserve ratio、sculpt interval 或任务权重。

第三，尝试视觉连接层或视觉侧 LoRA：

优先考虑：

- aligner/projector 相关模块；
- visual merger；
- vision attention 的少量 LoRA。

不建议一开始直接全量解冻 ViT，因为显存成本和过拟合风险更高。更稳妥的路线是先让视觉到语言的连接层具备适配能力。

第四，最后再考虑 `30% targeted`：

如果 20% 的任务级 targeted、格式约束和训练侧配置都验证完后仍有明显差距，再扩大到 30%。否则直接训练 30% 很可能只是增加成本，不能解释问题来源。

### 预期收益

如果 localization 的瓶颈确实来自视觉 grounding，训练侧改进应表现为：

- localization 分数明显高于 `79.79/80.13`；
- bbox IoU 提升，而不是只提升 no-object 判断；
- `Ours 错但 Traffic Full 对` 的 localization 样本减少；
- counting 对小目标、多目标场景也有一定改善。

### 验证方式

训练后重点看三类结果：

1. 总表中的 localization 分数；
2. `focus_wrong_reference_right.csv` 中 localization 样本数量；
3. bbox 解析细节，包括 `pred_boxes`、`matched`、`ious` 和 `iou50`。

如果 localization 分数提升但 bbox IoU 没提升，需要警惕 scorer 或 no-object 样本占比导致的虚假提升。

## 推荐执行顺序

建议不要直接训练 30% targeted，而是按以下顺序推进：

1. 构建 QA-level / task-aware targeted 20% 数据；
2. 训练前统计 QA 分布，确认 counting/localization 占比真的提高；
3. 使用相同训练配置训练 `20% task-aware targeted + LoRASculpt`；
4. 推理时使用 counting/localization-only 格式约束；
5. 与 `15% merged`、`20% targeted`、`Traffic Full` 比较；
6. 如果 localization 仍弱，再做训练分辨率对齐和视觉连接层 LoRA；
7. 最后才考虑 `30% targeted` 或更高 LoRA rank。

这个顺序可以最大程度降低试错成本，并且每一步都能回答一个明确问题：

| 步骤 | 回答的问题 |
|---|---|
| QA-level targeted | 是不是任务监督比例不足？ |
| 选择性格式约束 | 是不是输出格式导致低估？ |
| 训练分辨率对齐 | 是不是小目标视觉信息不足？ |
| 视觉侧/连接层 LoRA | 是不是视觉 grounding 能力不足？ |
| 30% targeted | 数据量扩大是否仍有边际收益？ |

## 论文中可形成的创新点

如果后续实验验证有效，可以把当前工作从“简单增加 targeted 数据”提升为更完整的方法贡献：

1. **Task-aware targeted data expansion**

   不只按场景选择数据，而是结合模型短板，提高 counting/localization 等关键任务的训练监督比例。

2. **Task-specific inference formatting**

   针对自动评估敏感任务设计最小格式约束，在不影响 recognition 的前提下提升 counting/localization 的可评分性。

3. **Grounding-oriented adaptation analysis**

   通过格式约束、训练分辨率、视觉连接层 LoRA 等实验区分“语言输出问题”和“视觉 grounding 问题”，让论文不只是报分数，而是解释能力瓶颈。

4. **Data-efficient traffic VLM adaptation**

   最终论文主线应强调：在不使用全量 MITS 数据的情况下，通过高价值数据选择、任务级 targeted expansion 和高效 LoRA 适配，显著提升交通监控 VQA 能力。
