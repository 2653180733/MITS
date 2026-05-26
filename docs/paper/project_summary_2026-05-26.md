# MITS 项目全面总结

日期：2026-05-26

---

## 1. 项目概述

本项目面向基于 **Qwen2.5-VL-7B-Instruct** 的 **MITS（多模态智能交通监控）场景领域适应**研究，目标是在单卡条件下构建"数据筛选 → 高效微调 → 抗遗忘"的联合优化框架。

**核心思路**：以 MITS 任务结构为核心，用任务感知数据筛选（Task-aware ScalSelection）控制学习内容，用 QLoRA 单卡注入交通领域知识，用 LoRASculpt 约束 LoRA 更新方向，最终优化领域性能、数据成本和通用能力保持率之间的平衡。

**工程边界**：

- 工作目录：`D:\Code\PyCode\MITS`
- MITS 数据集路径（AutoDL）：`/root/autodl-tmp/data/dataset`
- 模型路径（AutoDL）：`/root/autodl-tmp/Qwen/Qwen2.5-VL-7B-Instruct`
- 本地 4070S 用于小样本预览和代码验证，AutoDL 用于特征提取、训练和大规模推理

---

## 2. 数据集与数据流

### 2.1 MITS 数据集结构

```
dataset/
├── images/<shard>/<scene>/images/<sample>.jpg
└── vqas/<shard>/<scene>/integratedinput/<sample>.json
    或 vqas/<shard>/<scene>/integratedinput_<subscene>/<sample>.json
```

每个 JSON 样本包含：

- `id`：原始样本 ID
- `image`：相对图像路径
- `basecaption` / `optimizedcaption`：场景描述对话
- `foregroundqa`：前景目标识别/计数/定位问答
- `backgroundqa`：背景场景分析（道路类型、天气、光照等）
- `reasoningqa` / `eventreasoningqa`：事件推理问答
- `llmqa`：LLM 生成的补充问答
- `baselabel` / `categorylabel`：场景和类别标签（含事故、烟火、抛洒、施工、天气等长尾标签）

### 2.2 数据流水线（7 步）

```
原始 MITS 数据
  → (1) build_mits_index.py      构建全量 JSONL 索引（scene/shard/task_tags/rare_tags）
  → (2) convert_mits_to_sharegpt.py  转换为 ShareGPT 多轮格式 + QA 质量筛选
  → (3) feature_extract_sft.py   特征提取（hybrid_meta 或 qwen_attention）
  → (4) cur.py                    CUR/SVD 重要性评分
  → (5) select_mits_subset.py    按 scene 分组 + CUR + task bonus + rarity bonus 筛选
  → (6) convert_mits_to_sharegpt.py  导出筛选后的训练集 JSONL
  → (7) swift sft + LoRASculpt   QLoRA 训练
```

#### 步骤 (2)：ShareGPT 转换与 QA 筛选

单张 MITS 图片可能包含上百到上千条 QA。转换时做质量筛选和数量压缩：

| 筛选阶段 | 操作 |
|---|---|
| 排除非 QA 字段 | baselabel、categorylabel、*_task_type 等元数据字段 |
| 质量过滤 | 去空值、去过短（Q<8字/A<3字）、去过长（Q>512字/A>1024字）、去低信息答案（yes/no/unknown 等） |
| 去重 | 去除完全相同的 QA 对 |
| Balanced 均衡抽取 | 按 task 类型 round-robin 轮流抽取，每任务最多 8 条，每图最多 32 条 |
| 首次插入 `<image>` token | 第一条 user message 自动添加 `<image>\n` 前缀 |

**关键设计决策**：全量建索引 → 全量转 ShareGPT → 全量提特征（只做一次）→ 按 scene 分组筛选 → 合并导出训练集。避免小场景（如 firesmoke 仅 89 张测试图）被大场景淹没。

#### 步骤 (3)：特征提取（两种模式）

| 项目 | `qwen_attention` (baseline) | `hybrid_meta` (默认) |
|---|---|---|
| attention 计算 | 是（output_attentions=True） | 否（sdpa） |
| 特征来源 | user→vision attention + hidden | vision hidden + QA text hash + meta hash |
| 显存占用 | 高（[batch, heads, seq, seq]） | 低 |
| 速度 | 慢 | 快 |
| 场景信息注入 | 间接依赖 CUR | 直接写入表示向量 |

#### 步骤 (5)：核心筛选公式

```
final_score = cur_score + λ_task × task_bonus + λ_rare × rarity_bonus
```

默认参数：`λ_task=0.2`, `λ_rare=0.1`, `group_by=scene`, `min_task_fraction=0.05`。

筛选在 **每个 scene 内部独立进行**，保证小场景（事故 174 张、烟火 89 张）不会在全局排序中被大场景（person_vehicle 313 张）淹没。

---

## 3. 方法论

### 3.1 三个核心模块

**Task-aware ScalSelection（任务感知数据筛选）**：

- 基于 CUR 矩阵分解计算每个样本的几何重要性
- 加入逆向频率任务覆盖奖励（task_bonus），保证五类任务均被覆盖
- 加入稀有类别奖励（rarity_bonus），提升事故/烟火/抛洒/施工/恶劣天气等长尾事件的采样权重
- 按 scene 分组独立筛选，避免小场景样本被淹没

**QLoRA（量化低秩适配）**：

- int4 量化 + bf16 计算，单卡 24GB 显存可运行
- LoRA rank=16, alpha=32, dropout=0.05
- target_modules：语言模型全部线性层（q/k/v/o/gate/up/down_proj）
- 不微调 vision encoder，保证视觉特征稳定性
- 训练配置：per_device_batch=1, gradient_accumulation=16, effective_batch=16
- 学习率 1e-4，cosine scheduler，1 epoch

**LoRASculpt（抗遗忘雕塑）**：

- 每隔 `sculpt_interval=300` optimizer steps 执行一次
- 将 LoRA 参数按绝对值排序，仅保留 top-k（preserve_ratio=0.10），其余归零
- 雕塑目标：q_proj、k_proj、v_proj、mm_projector 等注意力相关模块
- 目的：削减低幅值参数更新，抑制对原始多模态能力的破坏

**Rank-Sculpt Co-design**：

| LoRA Rank | Preserve Ratio | 说明 |
|---|---:|---|
| 8 | 0.15 | 显存较紧，更保守保留 |
| 16 | 0.10 | 默认主实验配置 |
| 32 | 0.05 | 显存较充足，需要更强雕塑 |

### 3.2 评估体系

**MITS 五类领域能力**：

| 能力 | 符号 | 自动/裁判 | 评分方式 |
|---|---|---|---|
| Object/Event Recognition | S_recog | 自动 | Yes/No 匹配 + 精确文本匹配 |
| Object Counting | S_count | 自动 | `max(0, 1 - abs(pred-gt)/max(gt,1))` |
| Object Localization | S_loc | 自动 | 预测框与 GT 框贪心 IoU 匹配 |
| Background Analysis | S_background | 裁判 | DeepSeek-R1 等 0-1 语义评分 |
| Event Reasoning | S_reasoning | 裁判 | 事实一致性与语义一致性 0-1 打分 |

**MITS Average** = mean(S_background, S_recog, S_count, S_loc, S_reasoning)

**通用能力保持率**（计划中，尚未执行）：

```
Retention = fine_tuned_score / original_qwen_score × 100%（MMBench / MME / SEED-Bench）
```

---

## 4. 代码组织

```
D:\Code\PyCode\MITS\
├── src/mits_pipeline/              # 核心库（5 个模块）
│   ├── mits_io.py                  #   MITS I/O、ShareGPT 转换、QA 过滤（663 行）
│   ├── selection.py                #   任务/稀有度感知子集筛选（228 行）
│   ├── eval_utils.py               #   自动评分、bbox 匹配、Judge 输出（559 行）
│   ├── lorasculpt.py               #   LoRASculpt 参数雕塑（通用版，104 行）
│   └── lowlight.py                 #   低照度图像检测与增强（147 行）
├── third_party/
│   ├── ScalSelect/                 #  CUR 筛选引擎 + MITS 工具链
│   │   ├── mits_tools/             #    流水线工具（11 个脚本）
│   │   └── scripts/               #    特征提取、CUR 打分
│   ├── LoRASculpt/                 #  LoRASculpt 参考实现 + Qwen 适配
│   │   └── mits_qwen/             #    Qwen 专属 Trainer 回调
│   └── UnicomBenchmark/           #  通用 benchmark（待用）
├── scripts/                        # 训练/评估 Shell 脚本
│   ├── run_swift_sft_lorasculpt_15_logged.sh
│   ├── run_mits_eval_base_traffic_ours.sh
│   ├── run_mits_eval_base_vs_lorasculpt.sh
│   └── swift_lorasculpt_plugin.py
├── configs/
│   └── qwen25vl_qlora_defaults.json
├── docs/
│   ├── paper/                      # 论文框架、实验流程图
│   ├── code/                       # 各模块改造文档（11 篇）
│   └── summery_2026-05-15.md      # 5/15 工作总结
└── swanlog/                        # 训练日志
```

---

## 5. 当前实验进展

### 5.1 已完成：一次完整实验循环

**训练**（2026-05-23，AutoDL）：

| 配置项 | 值 |
|---|---|
| 基座模型 | Qwen2.5-VL-7B-Instruct |
| 训练数据 | MITS 15% 子集，每图最多 32 条 QA，balanced 均衡 |
| 微调方式 | QLoRA (int4) + LoRA rank=16 |
| LoRASculpt | interval=300, preserve_ratio=0.10 |
| 训练步数 | 1540 steps（1 epoch） |
| 最终 checkpoint | checkpoint-1540 |

**评估**（2026-05-25，AutoDL）：

- 评估集：1000 张测试图 × 每图 5 条 QA = 5000 条
- 三方对比：Base Qwen vs Traffic Full vs Ours (15% + LoRASculpt)
- 场景分布：person_vehicle(313) > accident(174) > spill(138) > construction(123) > weather(107) > firesmoke(89) > jam(56)

### 5.2 当前结果

| 指标 | Base (原始) | Traffic Full (全量微调) | Ours (15%+Sculpt) | Δ Ours-Base | Δ Ours-Traffic |
|---|---:|---:|---:|---:|---:|
| Recognition | 82.90 | 98.70 | 95.30 | +12.40 | −3.40 |
| Counting | 80.21 | 95.00 | 85.00 | +4.79 | −10.00 |
| Localization | 59.82 | 85.77 | 79.47 | +19.65 | −6.30 |
| Background | — | — | — | — | — |
| Reasoning | — | — | — | — | — |
| **Automated Avg** | **74.31** | **93.15** | **86.59** | **+12.28** | **−6.57** |

**关键发现**：

1. Ours 相比原始 Qwen 在三个自动评分任务上平均提升 **12.28 个百分点**，localization 提升最大（+19.65），counting 提升最小（+4.79）
2. Ours 用 **15% 数据达到了 Traffic Full 的 93.0%**（86.59/93.15），但 counting 差距最大（−10.00），说明 15% 子集中计数类 QA 训练不够
3. Background 和 Reasoning（各 1000 条，共 2000 条）需要 Judge 模型评分后才能计算完整的 MITS Average
4. **注意**：Traffic Full 是全量参数微调（7B 参数全改），Ours 仅微调 LoRA 适配器（约 1% 参数），两者不在同一层级。论文框架定义的上界应为 **Full QLoRA（100% 数据 + QLoRA）**，尚未执行

### 5.3 已发现并修复的问题

**Bug 1：Counting 评分 `_NO_OBJECT_PATTERNS` 遗漏**

`eval_utils.py` 中的 `_NO_OBJECT_PATTERNS` 无法匹配 "does not contain" / "does not show" 等常见否定表达，导致 counting ground truth 为"图中无对象"时无法评分。

修复：新增 `r"\b(?:does|do|did) not contain\b"` 和 `r"\bdoesn't contain\b"`。但 "does not show" 仍需补上。

**Bug 2：数据质量问题**

- 少量 localization 样本的 ground truth 不含坐标框（纯文字描述），导致评分时 `error=no_ground_truth_bbox`
- 少量 counting 样本的 ground truth 不含数字（如 "The types of vehicles are: ['bike', 'car', 'truck']" 被错误标注为计数问题）

---

## 6. 待完成工作

### 6.1 必做实验（论文必需）

| 优先级 | 实验 | 说明 |
|---|---|---|
| **P0** | **Full QLoRA**（100% 数据） | 论文框架定义的领域性能上界，不加 LoRASculpt |
| **P0** | **Judge 评分**（Background + Reasoning） | 补全五任务分数，计算完整 MITS Average |
| **P0** | 20% / 25% 子集 + QLoRA + Sculpt | 寻找接近 Full QLoRA 的最低数据比例 |
| P1 | 5% / 10% / 30% 子集 | 绘制比例-性能曲线 |
| P1 | 通用能力评估（MMBench / MME / SEED-Bench） | 计算 Retention / Forgetting |

### 6.2 消融实验

| 优先级 | 实验 | 验证内容 |
|---|---|---|
| P1 | Random 15% + QLoRA | 验证 task-aware selection 收益 |
| P1 | ScalSelect (qwen_attention) 15% + QLoRA | 验证 hybrid_meta 特征后端收益 |
| P2 | Ours 15% 不加 LoRASculpt | 验证 LoRASculpt 抗遗忘贡献 |
| P2 | 不同 preserve_ratio (0.05/0.15/0.20) | 验证雕塑强度影响 |
| P2 | 不同 LoRA rank (8/32) | 验证 rank-sculpt co-design |
| P3 | QA 数量消融（每图 16/32/64 条） | 验证 QA 密度影响 |
| P3 | 低照度增强 | 验证 lowlight augmentation 收益 |

### 6.3 后续改进方向

1. **提升 counting 性能**：当前 counting 差距最大（−10.00 vs Traffic Full）。可尝试提高每图 counting 类 QA 上限（`max-pairs-per-task` 从 8 提到 12）、或提高数据比例到 20-25%
2. **补修 "does not show" 评分 bug**：在 `_NO_OBJECT_PATTERNS` 中加入 `r"\bdoes not show\b"` 和 `r"\bdoes not (?:show|depict|display)\b"`
3. **Judge 评分流程**：`judge_background_reasoning.jsonl` 已生成（三模型各 2000 条），需用 DeepSeek-R1 或 GPT-4 等裁判模型逐条打分，汇总后重新计算完整 MITS Average
4. **训练数据标记追踪**：当前 eval set 构建时从总池 139604 个候选中排除了训练集已有的 24636 个样本（by original_id 和 image_key），确保评估无泄漏

### 6.4 技术债务

- 本地 4070S 未安装 torch，无法在本地跑特征提取和预测脚本
- Windows ↔ AutoDL 路径同步需手动处理
- 缺少一站式实验管理脚本（当前需手动串联 7+ 个步骤）

---

## 7. 运行参考

### 7.1 常用命令

**全量流水线（AutoDL 一键）**：

```bash
cd /root/autodl-tmp/MITS
bash third_party/ScalSelect/mits_tools/run_mits_pipeline_autodl.sh
```

**训练**：

```bash
# 默认 15% + LoRASculpt
RATIO=15 ENABLE_SWANLAB=1 bash scripts/run_swift_sft_lorasculpt_15_logged.sh

# 调整雕塑强度
LORASCULPT_PRESERVE_RATIO=0.15 bash scripts/run_swift_sft_lorasculpt_15_logged.sh

# 仅 QLoRA（不加 Sculpt）
# 去掉 --external_plugins 参数即可，或手动修改脚本
```

**三方评估**：

```bash
export MODEL_PATH=/root/autodl-tmp/Qwen/Qwen2.5-VL-7B-Instruct
export TRAFFIC_MODEL=/root/autodl-tmp/zhaokaikai/Qwen2.5-VL-7B-Instruct-Traffic
export OURS_ADAPTER=/root/autodl-tmp/data/train_outputs/.../checkpoint-1540

# 小样本测试
EVAL_LIMIT=200 MAX_PIXELS=1048576 bash scripts/run_mits_eval_base_traffic_ours.sh

# 全量评估
bash scripts/run_mits_eval_base_traffic_ours.sh
```

**单步操作**（AutoDL 路径已固化）：
```bash
python third_party/ScalSelect/mits_tools/build_mits_index.py --limit 0 --allow-full-scan
python third_party/ScalSelect/mits_tools/convert_mits_to_sharegpt.py --max-pairs-per-sample 32 --qa-filter balanced --max-pairs-per-task 8
python third_party/ScalSelect/scripts/feature_extract_sft.py --feature-mode hybrid_meta
python third_party/ScalSelect/scripts/cur.py
python third_party/ScalSelect/mits_tools/select_mits_subset.py --ratio 20
```

### 7.2 当前环境变量约定

| 变量 | AutoDL 默认值 |
|---|---|
| `DATASET_ROOT` | `/root/autodl-tmp/data/dataset` |
| `MODEL_PATH` | `/root/autodl-tmp/Qwen/Qwen2.5-VL-7B-Instruct` |
| `WORK_DIR` | `/root/autodl-tmp/data/outputs/full` |
| `TRAFFIC_MODEL` | `/root/autodl-tmp/zhaokaikai/Qwen2.5-VL-7B-Instruct-Traffic` |

---

## 8. 输出产物一览

当前 `D:\Code\PyCode\MITS_outputs\outputs\full\eval_test1000\` 下的完整输出结构：

```
eval_test1000/
├── mits_test_qas.jsonl              # 评估集（5000 条 QA）
├── mits_test_index.jsonl            # 评估集索引
├── mits_eval_set_summary.json       # 评估集构建统计
├── predictions/
│   ├── base_mits_test.jsonl         # Base Qwen 预测（5000 条）
│   ├── traffic_full_mits_test.jsonl # Traffic Full 预测（5000 条）
│   └── ours_lorasculpt_mits_test.jsonl  # Ours 预测（5000 条）
├── scores/
│   ├── base_mits_test/
│   │   ├── summary.json             # 评分汇总
│   │   ├── summary.csv              # 五任务分项 CSV
│   │   ├── per_sample_scores.jsonl  # 逐条评分明细
│   │   └── judge_background_reasoning.jsonl  # 待 Judge 评分的样本
│   ├── traffic_full_mits_test/
│   │   └── ...（同上）
│   └── ours_lorasculpt_mits_test/
│       └── ...（同上）
└── compare/
    ├── base_vs_traffic_vs_ours_mits_test.md   # 三方对比报告
    └── base_vs_traffic_vs_ours_mits_test.csv  # 三方对比数据
```
