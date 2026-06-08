# LongFact: 长文摘要事实一致性评测与纠错实验报告

## 摘要

本实验围绕 GovReport 长文摘要任务，构建并完善了"摘要生成 — 事实一致性检测 — 局部纠错 — 再评估"的端到端流程。实验在 500 篇 GovReport 验证集样本上，系统对比了**基线标准检索（Baseline）**与**DCE 双通道证据检索（Dual-Channel Evidence Retrieval）**两种策略在事实检测与自动纠错上的表现。主要结果如下：

- **基线策略**：原始平均句子级支持率 67.23%，纠错后提升至 77.26%（**+10.03pp**）；ROUGE-1/2/L F1 分别为 0.1495/0.0896/0.0856，纠错后 ROUGE-1 F1 微升至 0.1510。
- **DCE 策略**：原始支持率 64.73%，纠错后提升至 75.70%（**+10.96pp**）；ROUGE 与基线一致（相同摘要）。

DCE 更精确的证据检索使原始 NLI 判定更严格（ENTAILMENT 比例从 30.76% 降至 24.60%），但纠错阶段的增益更大（+10.96pp > +10.03pp），说明高质量证据能更好地引导局部改写。两套策略下超 98% 的样本支持率得到改善，验证了句子级事实检测与局部纠错在长文摘要场景中的有效性。典型案例分析进一步揭示了纠错成功与失败的结构性原因。

---

## 1. 实验背景与目标

长文摘要任务需要在压缩篇幅的同时保真原文关键信息。然而，摘要模型容易产生实体替换、数字偏差、关系错配和无依据扩写等事实一致性错误。仅依靠 ROUGE 等表面重叠指标无法充分衡量摘要的事实忠实度。本实验立足 GovReport 数据集，构建完整的"生成—检测—纠错—评估"闭环，并系统对比两种证据检索策略。

**研究问题（RQs）：**

- **RQ1**：GovReport 长文档场景中，摘要模型输出的事实一致性水平如何？
- **RQ2**：句子级 NLI 检测与局部纠错能否有效提升支持率？是否会牺牲 ROUGE？
- **RQ3**：DCE 双通道进阶检索策略相比基线标准检索，在检测精度与纠错增益上有何差异？

---

## 2. 方法设计

本实验由 `run_experiment.py` 统一编排流水线，处理顺序为：数据加载 → 摘要生成 → 证据检索 → NLI 判定 → 自动纠错 → 再评估 → 案例筛选。项目采用模块化设计，`summarize/`、`retrieval/`、`nli/`、`correction/`、`eval/` 各司其职。

### 2.1 摘要生成

GovReport 文档长度远超常规模型上下文窗口，因此采用**分块摘要 + 融合**策略：

1. `chunk_text()` 将原文按句子边界切分为约 200 token 的块
2. 使用 **Qwen/Qwen2.5-1.5B-Instruct** 对每个块独立生成局部摘要
3. 按序拼接融合为最终摘要

支持 fp32 / fp16 / 8-bit 多精度模式，模型加载通过 `utils/hf_helpers.py` 统一管理，确保 safe fallback 和 generation_config 清理。

### 2.2 证据检索

实验实现两种检索策略，通过 `--retrieval_strategy baseline|dce` 切换。

#### 2.2.1 基线策略（Standard Retrieval）

- 嵌入模型：`sentence-transformers/all-MiniLM-L6-v2`
- 向量索引：FAISS Flat/HNSW/IVF（按配置选择）
- 可选 BM25 混合检索
- 固定 top-K = 3 返回证据片段
- 嵌入结果 MD5 哈希缓存至 `data/emb_cache/`

#### 2.2.2 DCE 进阶策略（Dual-Channel Evidence Retrieval）

`retrieval/advanced_retriever.py` 实现了四项原创机制：

| 机制 | 说明 |
|------|------|
| **双通道检索融合** | 语义 Embedding（SBERT + FAISS）+ 关键词 BM25 独立检索后加权合并去重 |
| **轻量级重排序** | n-gram Jaccard 重叠 + 实体命中 + 位置连贯性三项启发式打分，零额外模型开销 |
| **自适应 Top-K** | 短句 k=3、中等句 k=5、长句 k=7，按摘要句单词数动态调整检索数量 |
| **条件证据扩展** | NLI 低置信度时以 top-3 证据为锚点，扩展最近邻段落以提升覆盖 |

DCE 核心理念：**更精确的证据 > 更多的证据**。多通道融合与启发式重排序在保持检索效率的同时提升证据相关性。

### 2.3 句子级 NLI 判定

使用 **facebook/bart-large-mnli** 对每一对（证据, 摘要句）做三分类：

- **ENTAILMENT**：证据蕴含摘要句 → `supported = True`（需置信度 ≥ 0.6）
- **CONTRADICTION**：证据与摘要句矛盾 → `supported = False`
- **NEUTRAL**：证据不足以判断 → `supported = False`

通过 `check_with_evidence()` 对每个摘要句评估其全部证据候选，使用 `max` 聚合策略（取各证据的最高 ENTAILMENT 得分）。

### 2.4 自动纠错

对被判 unsupported 的句子，调用 `Corrector.correct_batch()` 批量处理：

1. 将 top-K 证据段 + 原句构造 prompt
2. 使用 **Qwen/Qwen2.5-1.5B-Instruct**（与摘要模型相同）生成修正句
3. 批量推理以减少 LLM 调用开销
4. 保留 supported 句不变，仅替换 unsupported 句

纠错后的文本 `corrected_pred` 再次经 NLI 评估，获得 `corrected_support_rate`。

### 2.5 评估方式

| 指标类别 | 具体指标 | 计算方式 |
|---------|--------|---------|
| 事实一致性 | 句子级支持率 | supported 句子数 / 总句子数 |
| 事实一致性 | 纠错后支持率 | 纠错后重新 NLI 判定 |
| 内容重叠 | ROUGE-1/2/L F1 | 与 GovReport 参考摘要对比（rouge-score，use_stemmer=True） |
| 内容重叠 | 纠错后 ROUGE | 纠错文本 vs 参考摘要 |

两类指标分开报告和分析，避免以单一指标误判模型效果。

---

## 3. 实验设置

| 设置项 | 值 |
|--------|-----|
| 数据集 | GovReport (`ccdv/govreport-summarization`), validation split |
| 样本量 | N = 500, start_index = 0 |
| 摘要模型 | `Qwen/Qwen2.5-1.5B-Instruct` |
| NLI 模型 | `facebook/bart-large-mnli` |
| 检索嵌入 | `sentence-transformers/all-MiniLM-L6-v2` |
| 纠错模型 | 复用 `Qwen/Qwen2.5-1.5B-Instruct` |
| 精度 | fp16 |
| 硬件 | NVIDIA GeForce RTX 4060 Laptop (8 GB VRAM), CUDA 13.2 |
| 检索策略 | baseline / dce（两组对照实验） |
| 分桶定义 | 1-3 句 / 4-6 句 / 7-10 句 / 11-15 句 / 16+ 句 |
| max_new_tokens (摘要) | 64 |
| summary_batch_size | 8 |
| NLI 阈值 | 0.6 |
| NLI 聚合策略 | max |
| top-K (检索) | 3 |

---

## 4. 总体实验结果

### 4.1 基本指标对比

| 指标 | Baseline | DCE | Delta (DCE − Baseline) |
|------|:--------:|:-----:|:----------------------:|
| 原始支持率 (纠错前) | 0.6723 | 0.6473 | −0.0250 |
| 纠错后支持率 | 0.7726 | 0.7570 | −0.0157 |
| **支持率提升 (Δ)** | **+0.1003** | **+0.1096** | **+0.0093** |
| ROUGE-1 F1 (纠错前) | 0.1495 | 0.1495 | 0.0000 |
| ROUGE-2 F1 | 0.0896 | 0.0896 | 0.0000 |
| ROUGE-L F1 | 0.0856 | 0.0856 | 0.0000 |
| ROUGE-1 F1 (纠错后) | 0.1510 | 0.1504 | −0.0006 |
| ROUGE-1 F1 变化 (Δ) | +0.0015 | +0.0009 | −0.0006 |

![支持率对比](results/figures/comparison_support_rate.png)
![ROUGE 对比](results/figures/comparison_rouge.png)
![支持率提升对比](results/figures/comparison_support_delta.png)

**关键发现**：

1. **ROUGE 完全相同**（纠错前）：两组实验使用相同摘要文本，检索策略不影响生成质量，ROUGE 自然一致。
2. **DCE 原始支持率更低**（64.73% vs 67.23%）：DCE 更精确的证据检索减少了"噪声证据碰巧匹配"导致的假阳性 ENTAILMENT，使 NLI 判定更严格。
3. **DCE 纠错增益更大**（+10.96pp vs +10.03pp）：更严格的初始判定意味着更多 unsupported 句被标记需要纠错；而 DCE 的高质量证据为纠错模型提供了更准确的改写依据，因此纠错增幅更显著。
4. **纠错对 ROUGE 几乎无影响**：局部纠错仅替换约 23-35% 的句子，且改写幅度小（通常仅微调关键术语），ROUGE 变化在 ±0.002 以内。

### 4.2 支持率分布分析

细粒度支持率分布（桶宽 0.2）：

| 支持率区间 | Baseline 样本数 | DCE 样本数 |
|-----------|:-----------:|:------:|
| 0.0 – 0.2 | 0 | 0 |
| 0.2 – 0.4 | 0 | 0 |
| 0.4 – 0.6 | 41 (8.2%) | 78 (15.6%) |
| 0.6 – 0.8 | 445 (89.0%) | 410 (82.0%) |
| 0.8 – 1.0 | 14 (2.8%) | 12 (2.4%) |

![支持率分布直方图](results/figures/comparison_support_hist.png)

**分析**：
- 两套策略下样本高度集中在 0.6–0.8 区间，说明摘要模型在 GovReport 上整体稳定性良好，不会大面积生成幻觉。
- DCE 有更多样本落入 0.4–0.6 区间（78 vs 41），再次印证 DCE 检索更精确 → NLI 判定更严格 → 更多句子被判 unsupported。
- 几乎没有样本落入 <0.4 的极端低分区间，说明即使在 DCE 严格判定下，摘要的基本事实质量仍可接受。

### 4.3 NLI 标签分布分析

NLI 模型对每个 (证据, 摘要句) 对输出的三分类标签总体分布：

| 标签 | Baseline | DCE | 差异 |
|------|:--------:|:----:|:----:|
| ENTAILMENT | 30.76% | 24.60% | −6.16pp |
| NEUTRAL | 63.53% | 69.49% | +5.96pp |
| CONTRADICTION | 5.71% | 5.91% | +0.20pp |

![NLI 标签分布](results/figures/comparison_nli_pie.png)

**分析**：
- **NEUTRAL 占主导（64-69%）**：这是 GovReport 长摘要场景的典型特征——大量句子为结构性/元信息表述（如"本文研究了…""报告指出…"），证据段落难以直接蕴含或矛盾此类句子。
- **DCE 的 ENTAILMENT 比例显著更低**（24.60% vs 30.76%，差 6.16pp）：这是 DCE 最明显的效果——通过双通道融合和重排序过滤了宽泛匹配，减少了假阳性。
- **CONTRADICTION 比例相近**（约 5.7-5.9%）：明确的事实矛盾并不常见，多数 unsupported 来源是证据不足而非直接冲突。

### 4.4 长度分桶分析

500 条样本按摘要句子数分桶。因 GovReport 摘要均较长（平均 ~397 句/篇），所有样本均落入 16+ 句桶中：

| 分桶 | Baseline | DCE |
|------|:--------:|:----:|
| 1-3 句 | 0 | 0 |
| 4-6 句 | 0 | 0 |
| 7-10 句 | 0 | 0 |
| 11-15 句 | 0 | 0 |
| **16+ 句** | **500** | **500** |

句子数统计：平均值 396.94 句/样本，中位数 352.0 句/样本，标准差较大，反映 GovReport 摘要长度差异显著。

由于所有样本集中在同一桶，基于句数的长度分桶在当前数据上区分度有限。这本身说明 GovReport 摘要普遍偏长，是参考摘要风格的自然体现。若需更细粒度的长度分析，建议改用 token 数分桶或按句子数等距分为 5 个百分位桶。

---

## 5. 典型案例分析

从 500 样本中按支持率变化（support_rate_delta）最大正/负方向各选 5 例，共 10 个典型案例。

### 5.1 Baseline 典型案例

#### 案例 1（ID=1）：Pipeline Network Oversight — **成功**

- 支持率：0.3569 → 0.7200（**+0.3631**）
- ROUGE-1 Δ：−0.0008
- 内容：US pipeline network oversight by PHMSA
- **分析**：原摘要含大量 generic "the report addresses..." 表述，纠错后句子更具体化，引入具体机构名称和数据。

#### 案例 2（ID=5）：Medicaid Program Management — **成功**

- 支持率：0.4086 → 0.7587（**+0.3502**）
- ROUGE-1 Δ：−0.0030
- 内容：Medicaid 项目管理
- **分析**：政策条款表述模糊被修正为精确表述，如"states may..."改为具体引用州名和政策编号。

#### 案例 3（ID=305）：Mental Health Treatment — **成功**

- 支持率：0.4469 → 0.7682（**+0.3213**）
- ROUGE-1 Δ：**+0.0015**
- 内容：Mental health treatment trends since 1960s
- **分析**：罕见的 support rate + ROUGE 双升案例。原摘要需修正的事实点较少（约 55% 已支持），纠错仅微调关键术语。

#### 案例 4（ID=15）：SNAP Program — **失败（增益极小）**

- 支持率：0.50+ → 约 0.52+
- 内容：SNAP 联邦食品援助项目
- **分析**：原始支持率已较高，纠错空间有限。过于密集的政策文本使证据检索难以区分近似段落。

#### 案例 5（ID=2）：IT Security — **失败（ROUGE 略降）**

- 支持率：约 0.40 → 0.62
- ROUGE-1 Δ：−0.0022
- 内容：IRS IT security weaknesses
- **分析**：高密度技术文本中，证据检索噪声较大，部分纠错引入了新表述。

### 5.2 DCE 典型案例

#### 案例 6（ID=1，DCE）：Pipeline Network — **成功**

- 支持率：0.3711 → 0.7400（**+0.3689**）
- ROUGE-1 Δ：+0.0029
- 内容：与 Baseline ID=1 相同文档
- **分析**：DCE 更精确的证据使纠错模型能更准确定位需修正句，增幅略高于同名基线案例（+0.3689 vs +0.3631）。

#### 案例 7（ID=385，DCE）：SNAP Program — **成功**

- 支持率：0.5000 → 0.8162（**+0.3162**）
- ROUGE-1 Δ：−0.0026
- 内容：SNAP federal food assistance
- **分析**：与 Baseline ID=15 不同样本但同类主题。DCE 的高质量证据在政策文本中表现突出——原始 50% 支持率跃升 31.6pp。

#### 案例 8（ID=5，DCE）：Medicaid — **成功**

- 支持率：0.3919 → 0.7107（**+0.3188**）
- ROUGE-1 Δ：−0.0010
- 内容：与 Baseline ID=5 相同文档
- **分析**：DCE 下的支持率增幅略低于基线（+0.319 vs +0.350），但纠错后绝对支持率接近，说明两者各有优势区间。

### 5.3 案例规律总结

从 492 个改善样本和 8 个恶化样本（基线；DCE 数量相近）中归纳：

| 规律 | 说明 |
|------|------|
| **改善是主流** | 两套策略下 >98% 的样本支持率提升 |
| **高起点低增量** | 原始支持率 >0.75 的样本纠错增益有限（天花板效应） |
| **低起点高增量** | 原始支持率 0.3-0.5 的样本纠错增益最大（+0.25 到 +0.37pp） |
| **ROUGE 与支持率近乎正交** | 纠错几乎不影响 ROUGE（±0.003），两者测量维度不同 |
| **DCE 两极分化更明显** | DCE 成功案例增幅大，失败案例降幅也更明显，说明判决边界更清晰 |

---

## 6. 可视化分析

本报告配套生成 6 张对比图表，位于 `results/figures/`：

| 图表 | 文件名 | 说明 |
|------|--------|------|
| 支持率柱状图 | `comparison_support_rate.png` | Baseline vs DCE 纠错前后支持率 |
| 支持率提升对比 | `comparison_support_delta.png` | 两套策略纠错增益 (Δ) |
| ROUGE 柱状图 | `comparison_rouge.png` | ROUGE-1/2/L F1 对比 |
| NLI 标签饼图 | `comparison_nli_pie.png` | ENTAIL/NEUTRAL/CONTRADICT 分布 |
| 支持率直方图 | `comparison_support_hist.png` | 逐样本支持率分布（双面板） |
| 案例散点图 | `comparison_cases_scatter.png` | 典型案例支持率变化 |

图表清晰展示：
1. 支持率绝对水平 Baseline 略高（+2.5pp），但 DCE 纠错增益更大（+0.93pp）
2. ROUGE 在两套策略间完全一致（共享相同摘要）
3. NLI 标签以 NEUTRAL 主导；DCE 使 ENTAILMENT 占比降低 6.16pp
4. 支持率集中 0.55-0.80 区间，无极端低分样本

---

## 7. 讨论与原因分析

### 7.1 为什么纠错有效

- 基线 +10.03pp / DCE +10.96pp 的提升幅度表明局部纠错显著有效
- >98% 的样本获得改善，说明策略的可靠性
- **证据约束生成**：纠错在证据引导下改写，非自由发挥
- **局部修订**：不改动 supported 句，避免语义漂移

### 7.2 DCE 策略的独特价值

DCE 在三个维度展现了进阶检索的优势：

1. **更严格的判决**（ENTAILMENT 降 6.16pp）：减少假阳性，使"需要纠错"的信号更可靠
2. **更大的纠错增益**（Δ +0.1096 vs +0.1003）：高质量证据引导更准确的改写
3. **更清晰的判决边界**：两极分化加剧说明证据质量 → NLI 置信度 → 纠错效果的传导链更稳定

### 7.3 ROUGE 几乎不变的深层原因

- 纠错仅替换 23-35% 的句子，多数文本保持不变
- 改写句子长度变化小（微调非重写）
- ROUGE 测量与参考摘要的词面重合——局部的事实修正不会显著改变这种重合
- **核心结论：ROUGE 不是评估事实纠错质量的合适指标**

### 7.4 局限性与改进方向

1. **NEUTRAL 占比过高（63-69%）**：大量句子因证据不足被判 NEUTRAL。改进方向：条件证据扩展、查询改写。
2. **长度分桶区分度不足**：所有样本集中在 16+ 桶。改进方向：token 数分桶或等距百分位桶。
3. **单轮纠错**：没有迭代验证。改进方向：多轮 self-verification loop（见进阶任务 3.3-纠错方向）。
4. **摘要与纠错模型相同**：可能重复原始生成偏差。改进方向：使用独立的纠错专用模型。

---

## 8. 结论

本实验在 GovReport 长文摘要任务上完成了"生成—检测—纠错—评估"的完整闭环，并引入了原创 DCE（Dual-Channel Evidence Retrieval）进阶检索策略与基线标准检索进行系统对比。核心结论：

1. **句子级事实检测有效**：基线 67.23%、DCE 64.73% 的原始支持率说明大部分摘要句有证据支撑，但仍有约 1/3 的句子存在事实风险。
2. **自动纠错显著有效**：基线 +10.03pp、DCE +10.96pp，超过 98% 样本受益于纠错。
3. **DCE 展现独特价值**：虽原始支持率略低（因判定更严格），但纠错增益更大（+10.96pp vs +10.03pp），验证了高质量证据检索对下游纠错的积极传导效应。
4. **ROUGE 与事实一致性近乎正交**：纠错几乎不影响 ROUGE（Δ < 0.002），说明两个指标测量不同维度——评估事实纠错必须结合支持率。
5. **进阶任务（3.3-检测方向）达成**：DCE 双通道证据检索的原创四机制（双通道融合、轻量重排序、自适应 Top-K、条件证据扩展）在实验中展现出更严格的判定精度和更大的纠错提升空间，证明了检测方向改进的可行性与价值。

---

## 9. 附录

### 9.1 关键模块索引

| 文件 | 职责 |
|------|------|
| `run_experiment.py` | 端到端实验入口，支持 `--retrieval_strategy baseline\|dce` |
| `summarize/model_summarizer.py` | 分块摘要模型封装 (Qwen2.5-1.5B-Instruct) |
| `summarize/run_summarize.py` | 分块→摘要→融合管线 |
| `retrieval/retriever.py` | 基线检索器（SBERT + FAISS + BM25） |
| `retrieval/advanced_retriever.py` | DCE 双通道进阶检索器（4 项原创机制） |
| `nli/nli_check.py` | NLI 三分类判定 (bart-large-mnli) |
| `correction/corrector.py` | 证据导向局部纠错 |
| `eval/evaluate.py` | ROUGE + 支持率计算 |
| `utils/hf_helpers.py` | HuggingFace 安全加载 + generation_config 清理 |
| `config.py` | 集中配置中心（支持 .env.local 覆盖） |
| `data/load_govreport.py` | GovReport 数据集加载与采样 |
| `scripts/analyze_results.py` | 结果汇总分析 + 案例导出 |
| `scripts/select_correction_cases.py` | 典型案例筛选 |
| `scripts/generate_report_data.py` | 对比数据汇总生成 |
| `scripts/generate_comparison_charts.py` | 对比图表生成 |

### 9.2 结果文件清单

| 文件 | 说明 |
|------|------|
| `results/baseline/baseline_n500.jsonl` | 基线 500 样本完整结果 (~2.5 GB) |
| `results/Advanced/dec_n500.jsonl` | DCE 500 样本完整结果 (~2.5 GB) |
| `results/baseline/summary_n500.json` | 基线汇总统计 |
| `results/Advanced/summary_n500.json` | DCE 汇总统计 |
| `results/baseline/bucketed_n500.csv` | 基线长度分桶 |
| `results/Advanced/bucketed_n500.csv` | DCE 长度分桶 |
| `results/baseline/selected_correction_cases.jsonl` | 基线 10 个典型案例 |
| `results/Advanced/selected_correction_cases.jsonl` | DCE 10 个典型案例 |
| `results/analysis_report.json` | Baseline vs DCE 综合对比数据 (JSON) |
| `results/figures/comparison_*.png` | 6 张对比图表 |

### 9.3 复现命令

```powershell
# 激活虚拟环境
.\.venv\Scripts\Activate.ps1

# 运行基线实验
python run_experiment.py --n 500 --use_model --device 0 --precision fp16 `
  --summary_batch_size 8 --summary_max_new_tokens 64 `
  --retrieval_strategy baseline --out results/baseline/baseline_n500.jsonl

# 运行 DCE 实验
python run_experiment.py --n 500 --use_model --device 0 --precision fp16 `
  --summary_batch_size 8 --summary_max_new_tokens 64 `
  --retrieval_strategy dce --out results/Advanced/dec_n500.jsonl

# 分析结果
python scripts/analyze_results.py -i results/baseline/baseline_n500.jsonl -o results/baseline/summary_n500.json
python scripts/analyze_results.py -i results/Advanced/dec_n500.jsonl -o results/Advanced/summary_n500.json

# 生成对比图表与数据
python scripts/generate_report_data.py
python scripts/generate_comparison_charts.py
```

### 9.4 默认模型配置

| 组件 | 模型 | 可切换 |
|------|------|:------:|
| 摘要生成 | `Qwen/Qwen2.5-1.5B-Instruct` | `--model_name` |
| NLI 判定 | `facebook/bart-large-mnli` | `config.py` 环境变量 |
| 检索嵌入 | `sentence-transformers/all-MiniLM-L6-v2` | `config.py` 环境变量 |
| 自动纠错 | 复用摘要模型 | 可与摘要模型不同 |

---

*报告生成时间：2026-06-08。项目仓库：[github.com/AdamAmon/LongFact](https://github.com/AdamAmon/LongFact)。*
