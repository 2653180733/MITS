# MITS 三位一体领域适应方案

日期：2026-05-11

## 1. 基本约束

本文档面向基于 Qwen-2.5-VL-7B-Instruct 的 MITS 智能交通监控场景领域适应研究，目标是在单卡条件下构建“数据筛选 -> 高效微调 -> 抗遗忘”的联合优化框架。

工程边界如下：

- `D:\Desktop\Daily\Master\LLM\MITS` 仅作为只读参考代码来源，不在其中修改代码。
- 所有新增代码、适配脚本、实验配置和文档均放在当前工作区 `D:\Code\PyCode\MITS`。
- MITS 数据集路径为 `D:\Code\PyCode\multi_traffic\src\data\archives\dataset`。
- 数据集规模很大，分析格式时只允许读取前 100 条样本或 metadata，不做一次性全量加载或全量遍历。
- 所有路径拼接使用 `os.path.join`，保证 Windows 文件系统兼容。

## 2. 方案诊断

当前路线 ScalSelection + LoRA/QLoRA + LoRASculpt 的主要风险不在工程可行性，而在三者目标不完全一致：

- ScalSelection 原始目标是基于视觉表示几何结构选择核心样本，容易偏向视觉离群或高多样性样本，但不保证覆盖 MITS 的识别、计数、定位、背景分析和事件推理五类能力。
- LoRA/QLoRA 负责低成本注入交通领域知识，但如果训练子集任务分布失衡，模型会过拟合常见车辆、白天、晴天等高频模式。
- LoRASculpt 在参数层面缓解灾难性遗忘，但若数据筛选阶段没有显式约束通用能力和长尾任务，抗遗忘模块只能事后补救。

建议将论文主线从“模块拼接”改为：

**Retention-Constrained Traffic Core-set Adaptation**：以 MITS 任务结构为核心，用任务感知数据筛选控制学习内容，用 QLoRA 单卡注入交通领域知识，用 LoRASculpt 约束 LoRA 更新方向，最终优化领域性能、数据成本和通用能力保持率之间的平衡。

## 3. 轻量级创新点

### 3.1 Task-aware ScalSelection

将 MITS 原始 JSON 中的 `basecaption`、`foregroundqa`、`baselabel`、`categorylabel` 等字段转换为 ShareGPT 多轮格式，再在 CUR 重要性分数外加入任务覆盖约束。

核心目标：

- 保证 Recognition、Counting、Localization、Background、Reasoning 五类任务均被覆盖。
- 避免 10%-20% 子集只保留视觉多样但任务单一的样本。
- 保持 ScalSelection 的 training-free 特性，不额外增加模型训练成本。

### 3.2 Label-Rarity Balanced Core-set

对事故、烟火、抛洒、施工、恶劣天气、夜间等长尾交通事件提高采样权重。

建议最终样本分数为：

```text
final_score = cur_score + lambda_task * task_coverage_bonus + lambda_rare * rarity_bonus
```

默认参数：

- `lambda_task = 0.2`
- `lambda_rare = 0.1`
- CUR `sv_threshold = 0.9`

### 3.3 Forgetting-aware LoRA Sculpt

将 LoRASculpt 从单纯按 LoRA A/B 参数幅值保留，扩展为带通用能力保持约束的 LoRA 子空间雕塑。

实现思路：

- 每隔固定 optimizer steps 执行一次雕塑。
- 在少量通用保留集或通用 benchmark mini-dev 上估计通用损失变化。
- 优先保留对通用能力影响大的 LoRA 子空间，抑制破坏原始多模态能力的更新方向。

### 3.4 Rank-Sculpt Co-design

LoRA rank 与雕塑保留率联动，避免 rank 增大后领域参数过度覆盖原始模型能力。

推荐配置：

| LoRA Rank | AB Preserve Ratio | 适用场景 |
|---:|---:|---|
| 8 | 0.15 | 显存较紧，追求稳定 |
| 16 | 0.10 | 默认主实验配置 |
| 32 | 0.05 | 显存较充足，需更强雕塑 |

单卡主实验建议使用 QLoRA int4 + rank 16。

## 4. 单卡实验设计

### 4.1 主实验清单

| 实验组 | 数据比例 | 方法 | 目的 | 主要指标 |
|---|---:|---|---|---|
| Base | 0% | 原始 Qwen-2.5-VL-7B-Instruct | 原始能力基线 | MITS 五任务分数、MMBench/MME/SEED |
| Full-LoRA | 100% | 全量 MITS LoRA/QLoRA | 领域性能上界 | MITS Average |
| Random | 5/10/15/20/30% | 随机子集 + QLoRA | 数据筛选下界 | 与 Full-LoRA gap |
| ScalSelect | 5/10/15/20/30% | 原始 CUR 筛选 + QLoRA | 数据筛选 baseline | MITS Average、训练成本 |
| Task-aware | 5/10/15/20/30% | 任务感知 CUR + QLoRA | 验证任务覆盖收益 | 五任务分项 |
| Rarity-balanced | 10/15/20% | 任务感知 + 稀有类别平衡 | 验证长尾交通事件收益 | accident/firesmoke/spill |
| +LoRASculpt | 10/15/20% | 筛选 + QLoRA + LoRASculpt | 验证抗遗忘 | 通用保持率 |
| Rank Ablation | 15% | rank 8/16/32 | 寻找单卡最优 rank | MITS Average、显存、保持率 |
| Sculpt Ablation | 15% | preserve 5/10/15/20% | 寻找雕塑强度 | MITS Average、通用保持率 |

### 4.2 效益曲线

必须报告 5%、10%、15%、20%、30% 子集比例下的曲线：

| 横轴 | 纵轴 | 曲线 |
|---|---|---|
| 子集比例 | MITS Average Score | Random / ScalSelect / Task-aware / Ours |
| 子集比例 | 与 Full-LoRA gap | 目标 gap <= 3% |
| 子集比例 | 通用能力保持率 | 目标 retention >= 90% |

通用能力保持率定义：

```text
Retention = fine_tuned_score / original_qwen_score * 100%
Forgetting = 1 - fine_tuned_score / original_qwen_score
```

建议报告格式：

| Benchmark | 原始 Qwen | 微调模型 | Retention | Forgetting |
|---|---:|---:|---:|---:|
| MMBench | A | B | B/A | 1-B/A |
| MME | A | B | B/A | 1-B/A |
| SEED-Bench | A | B | B/A | 1-B/A |

## 5. 工程实现逻辑

### 5.1 MITS 转 ShareGPT

MITS 单条样本已确认包含 `id`、`image`、`baselabel`、`categorylabel`、`basecaption`、`foregroundqa` 等字段。转换时保留图像路径和多轮问答，并将标签摘要放入 `meta`。

```python
import os
import json


def mits_item_to_sharegpt(item, image_root):
    image_path = os.path.join(image_root, item["image"])

    messages = []
    qa_fields = [
        "basecaption",
        "foregroundqa",
        "backgroundqa",
        "reasoningqa",
        "optimizedcaption",
    ]

    first_user = True
    for field in qa_fields:
        if field not in item:
            continue

        turns = item[field]
        for i in range(0, len(turns) - 1, 2):
            question = turns[i]["value"]
            answer = turns[i + 1]["value"]

            if first_user and "<image>" not in question:
                question = "<image>\n" + question
            first_user = False

            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": answer})

    return {
        "id": item["id"],
        "messages": messages,
        "images": [image_path],
        "meta": {
            "baselabel": item.get("baselabel"),
            "categorylabel": item.get("categorylabel"),
        },
    }
```

只读取前 N 条样本的安全迭代器：

```python
import os
import json


def iter_mits_json(vqa_root, limit=None):
    count = 0

    for shard in sorted(os.listdir(vqa_root)):
        shard_dir = os.path.join(vqa_root, shard)
        if not os.path.isdir(shard_dir):
            continue

        for category in sorted(os.listdir(shard_dir)):
            cat_dir = os.path.join(shard_dir, category, "integratedinput")
            if not os.path.isdir(cat_dir):
                continue

            for name in sorted(os.listdir(cat_dir)):
                if not name.endswith(".json"):
                    continue

                path = os.path.join(cat_dir, name)
                with open(path, "r", encoding="utf-8") as f:
                    yield json.load(f)

                count += 1
                if limit is not None and count >= limit:
                    return
```

### 5.2 Task-aware ScalSelection

ScalSelect 原始实现要求 ShareGPT 输入格式。适配 MITS 时先构建轻量索引，不加载图像内容。

```python
def compute_task_tags(item):
    tags = set()

    if "foregroundqa" in item:
        tags.update(["recognition", "counting", "localization"])

    if "backgroundqa" in item:
        tags.add("background")

    if "reasoningqa" in item:
        tags.add("reasoning")

    category = item.get("categorylabel", [{}])[0]
    rare = []
    for key in ["accident", "firesmoke", "spill", "construction", "weather"]:
        if key in category:
            rare.append(key)

    return list(tags), rare


def combine_selection_score(cur_score, task_bonus, rarity_bonus,
                            lambda_task=0.2, lambda_rare=0.1):
    return cur_score + lambda_task * task_bonus + lambda_rare * rarity_bonus
```

### 5.3 LoRASculpt 接入 Qwen-2.5-VL

推荐 LoRA target modules：

```python
target_modules = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
```

雕塑接入点：

- 语言模型 attention：`q_proj`、`k_proj`、`v_proj`。
- 可选 MLP：`gate_proj`、`up_proj`、`down_proj`。
- 多模态桥接层：Qwen2.5-VL 中等价于 visual projector 或 multimodal merger 的线性层。
- 默认不雕塑 vision encoder 主干，避免显存和训练稳定性风险。

默认训练配置：

```text
load_in_4bit = true
bnb_4bit_compute_dtype = bfloat16
gradient_checkpointing = true
lora_rank = 16
lora_alpha = 32
lora_dropout = 0.05
AB_PRESERVE_RATIO = 0.10
sculpt_interval = 200-500 optimizer steps
```

基础雕塑逻辑：

```python
def sculpt_lora_params(model, preserve_ratio):
    for name, param in model.named_parameters():
        if "lora_" not in name:
            continue
        if not any(key in name for key in ["q_proj", "k_proj", "v_proj", "mm_projector"]):
            continue

        flat = param.data.abs().flatten()
        k = max(1, int(flat.numel() * preserve_ratio))
        threshold = flat.topk(k).values.min()

        mask = param.data.abs() >= threshold
        param.data.mul_(mask)
```

### 5.4 内存高效数据加载

实现原则：

- JSON 按样本懒加载。
- 图像只在 `__getitem__` 中打开。
- 不缓存 PIL Image。
- metadata 只保存轻量 JSONL 索引。
- Windows 路径统一使用 `os.path.join`。

```python
import json
import os
from PIL import Image
from torch.utils.data import Dataset


class MITSLazyDataset(Dataset):
    def __init__(self, index_file, image_root, processor):
        self.image_root = image_root
        self.processor = processor
        with open(index_file, "r", encoding="utf-8") as f:
            self.records = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]

        with open(rec["json_path"], "r", encoding="utf-8") as f:
            item = json.load(f)

        image_path = os.path.join(self.image_root, item["image"])
        image = Image.open(image_path).convert("RGB")

        sample = mits_item_to_sharegpt(item, self.image_root)
        return {
            "image": image,
            "messages": sample["messages"],
            "id": item["id"],
        }
```

## 6. MITS 论文评估方法

MITS 论文按五类能力评估模型：

| 能力 | 记号 | 评估方式 |
|---|---|---|
| Object/Event Recognition | `S_recog` | Yes/No 或类别识别准确率 |
| Object Counting | `S_count` | 基于预测数量与 GT 数量的相对误差 |
| Object Localization | `S_loc` | 预测框与 GT 框的 IoU |
| Background Analysis | `S_background` | 使用 DeepSeek-R1 等评分模型，对预测与标准答案做 0-1 语义评分 |
| Event Reasoning | `S_reasoning` | 使用评分模型做 0-1 事实一致性与语义一致性评分 |

Counting 可按论文语义实现为相对误差型得分：

```text
S_count = max(0, 1 - abs(pred_count - gt_count) / max(gt_count, 1))
```

总体分数：

```text
MITS Average = mean(
    S_background,
    S_recog,
    S_count,
    S_loc,
    S_reasoning
)
```

最终论文主表建议：

| Method | Data Ratio | Background | Recognition | Counting | Localization | Reasoning | Average | Retention |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen-2.5-VL | 0% | | | | | | | 100% |
| Full-LoRA | 100% | | | | | | | |
| ScalSelect | 15% | | | | | | | |
| Ours | 15% | | | | | | | |
| Ours | 20% | | | | | | | |

## 7. 默认实验假设

- 主实验以 15% 和 20% 子集为核心，5%、10%、30% 用于效益曲线。
- 单卡默认使用 QLoRA int4、bf16、梯度检查点、LoRA rank 16。
- 子集训练目标是 MITS Average 与 Full-LoRA 的 gap 控制在 3% 以内。
- 通用能力保持率以 MMBench、MME、SEED-Bench 平均 retention >= 90% 为合格线。
- MITS 论文 PDF 中部分公式文本提取存在格式缺失，Counting 按论文语义采用相对误差型分数。
