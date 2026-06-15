# TrustSearch — 消融实验设计与说明

## 核心思想

TrustSearch 的目标是训练一个 RAG 模型同时兼顾三个维度：

1. **性能 (Performance)**：答案正确率 (EM)
2. **可信度 (Trustworthiness)**：模型是否真正利用了检索证据，而非忽略证据靠记忆/编造回答
3. **成本 (Cost)**：检索次数尽可能少（会的题直接答，不会的才搜索）

### 关键洞察：性能与可信度不应该是 trade-off

反事实测试把"搜索后答对"的样本分为：
- `TRUE_TOOL`：替换文档中的答案为假实体后，模型答案**改变了** → 真正依赖了文档
- `PARAM_HALL`：替换后模型**照样答对** → 靠参数记忆，未使用文档

**`PARAM_HALL` 不是幻觉，而是"搜索冗余"**。如果把它惩罚为 0 或负分，会人为删除正确答案，导致 EM 下降——这是一个**人为制造的伪 trade-off**。

### Balanced 奖励设计（解决方案）

```
答案为空                → 0
不搜索 + 答对           → 1.0 + bonus (最省，鼓励会就直接答)
不搜索 + 答错           → -penalty (该搜不搜)
搜索 + 答错             → 0
搜索 + 答对             → 1.0 (性能保底，永不为 0)
                         + bal_ground × ground_signal  (接地加分，只加不减)
                         - w_cost × norm_cost          (成本轻罚)
```

- **答对永远是正分**（性能保底）→ 不会删正确答案
- **trust 是加分**：真用证据多拿分，靠记忆不加分但**不扣分**
- **成本独立扣项**：搜索越多轻微扣分

---

## 消融实验总览

### 主轴：固定"性能保底 + 成本项"框架，只替换 trust 信号

| 组 | 实验名 | trust 信号 | 成本项 | 设计目的 |
|---|---|---|---|---|
| **T0** | balanced −trust | 无 (ground=0) | ✓ | 基线：无 trust 加分，看 hall_rate 是否更高 |
| **T1** | balanced + cf | 反事实注入 | ✓ | 主方法：用反事实判断是否真用了证据 |
| **T2** | balanced + lexical | 词面接地 | ✓ | 替代：答案字面出现在文档中=接地（零成本） |
| **T3** | balanced + nli | 蕴含忠实度 | ✓ | 替代：judge 判断"证据是否支持答案"（最贴目标） |
| **A2** | balanced −cost | 反事实 | ✗ | 消融：去掉成本项，看 search/q 是否暴涨 |
| **A5** | cf_primary (artifact) | 反事实+惩罚PARAM_HALL | ✓ | **反例**：故意惩罚"靠记忆答对"→预期掉 EM |

### 已有参照基准（256 样本评测）

| 参照 | EM | trust@correct | hall_rate | search/q |
|---|---:|---:|---:|---:|
| base（未训练 Qwen2.5-3B-Instruct） | 0.207 | 0.930 | 0.02 | 1.97 |
| 旧 TrustSearch (legacy) step150 | 0.367 | 0.930 | 0.03 | 1.99 |
| baseline (EM-only) step250 | 0.441 | 0.966 | 0.04 | 3.00 |

---

## Trust 信号详解

### 1. 反事实注入 (Counterfactual Injection, cf)
- **方法**：将检索文档中的正确答案替换为假实体，让模型重新生成；如果答案改变，说明模型真正依赖了文档
- **优点**：因果性强，能精确区分"用了"vs"没用"
- **缺点**：奖励"被假文档带跑"（轻信检索），惩罚参数知识的鲁棒性
- **服务**：`/judge_batch` 接口

### 2. 词面接地 (Lexical Grounding, lexical)
- **方法**：检查规范化后的答案字符串是否出现在检索文档中
- **优点**：零成本、无需额外模型/服务、极快
- **缺点**：表面匹配≠真正使用，但胜在性价比
- **服务**：无需（本地字符串匹配）

### 3. 蕴含忠实度 (NLI Faithfulness, nli)
- **方法**：让 judge 模型判断"检索证据是否**支持**这个答案"（前提=docs，假设=answer）
- **优点**：正向衡量"有据可查"，不惩罚参数知识，不奖励"被假文档带跑"
- **缺点**：依赖 judge 模型质量
- **服务**：`/judge_faithfulness` 接口

### 4. 删证据因果 (Leave-One-Out Removal, removal)
- **方法**：将正确答案从文档中移除（而非替换为假实体），看模型是否还能答对
- **优点**：测因果依赖，无"奖励轻信"副作用
- **服务**：`/judge_removal` 接口（已实现，待后续实验）

---

## 实验详细配置

### 统一超参数

| 参数 | 值 | 说明 |
|---|---|---|
| 基座模型 | Qwen2.5-3B-Instruct | — |
| 算法 | GRPO | — |
| train_batch_size | 128 | — |
| val_batch_size | 64 | — |
| max_prompt_length | 4096 | — |
| max_response_length | 500 | — |
| max_turns (search) | 2 | 最多搜索 2 轮 |
| retriever topk | 3 | 每次检索返回 3 篇 |
| learning_rate | 1e-6 | — |
| KL loss | low_var_kl, coef=0.001 | — |
| n_agent (rollout samples) | 5 | — |
| val_before_train | True | step0 先验证 |
| test_freq | 25 steps | 每 25 步验证 |
| save_freq | 50 steps | 每 50 步保存 |
| 数据集 | NQ (Natural Questions) | train/test split |

### 各实验特有配置

#### T0: balanced −trust
```bash
ECO_TRUST_VARIANT=balanced
ECO_BAL_GROUND=0.0       # 无接地加分
ECO_W_COST=0.3
```

#### T1: balanced + cf (主方法)
```bash
ECO_TRUST_VARIANT=balanced
ECO_TRUST_SIGNAL=cf
ECO_BAL_GROUND=0.5       # 接地加分 0.5
ECO_W_COST=0.3
```

#### T2: balanced + lexical
```bash
ECO_TRUST_VARIANT=balanced
ECO_TRUST_SIGNAL=lexical
ECO_BAL_GROUND=0.5
ECO_W_COST=0.3
```

#### T3: balanced + nli
```bash
ECO_TRUST_VARIANT=balanced
ECO_TRUST_SIGNAL=nli
ECO_BAL_GROUND=0.5
ECO_W_COST=0.3
CF_JUDGE_URL="http://<nli_judge_host>:8001/judge_batch"  # 训练走扩展版judge
```

#### A2: balanced −cost
```bash
ECO_TRUST_VARIANT=balanced
ECO_TRUST_SIGNAL=cf
ECO_BAL_GROUND=0.5
ECO_W_COST=0.0           # 无成本项
```

#### A5: cf_primary (artifact / 反例)
```bash
ECO_TRUST_VARIANT=cf_primary
ECO_HALL_PENALTY=0.5     # PARAM_HALL -> -0.5
ECO_W_COST=0.2
```

---

## 这套消融要回答的核心问题

1. **T1 vs T0**：接地加分是否真的降低 hall_rate / 提升证据利用？
2. **T1 vs A2**：成本项对 search/q 和 oversearch 的控制效果？
3. **T1 vs T2 vs T3**：哪种 trust 信号在不牺牲性能的前提下**最能降幻觉**？
4. **A5 vs others**：验证"惩罚 PARAM_HALL 会掉 EM"（伪 trade-off 的实验证据）
5. **T1 vs legacy vs baseline**：新 balanced 设计能否在 **EM 追平 baseline** 的同时保持低 search/q？

---

## 代码结构

```
verl/trainer/main_ppo_eco.py   # 核心 reward manager（支持所有变体）
cf_judge_server.py             # Judge 服务（反事实/NLI/删证据三个接口）
eco_train_common.sh            # 训练公共入口
train_ts_balanced.sbatch       # T1 主方法训练脚本
train_ts_bal_nocost.sbatch     # A2 消融（−cost）
train_ts_bal_notrust.sbatch    # T0 消融（−trust）
train_ts_lexical.sbatch        # T2 词面接地
train_ts_nli.sbatch            # T3 蕴含忠实度
train_ts_cfprimary.sbatch      # A5 反例
start_cf_judge.sbatch          # 原版 judge 服务
start_cf_judge_b.sbatch        # 扩展版 judge（+faithfulness/removal）
start_retriever_*.sbatch       # 检索服务
eval_3dim_common.sh            # 三维评测公共脚本
run_eval-*.sbatch              # 评测脚本
```

---

## 三维评测指标

| 维度 | 指标 | 说明 |
|---|---|---|
| 性能 | EM (Exact Match) | 答案正确率 |
| 可信度 | trust@correct | 答对样本中"真用证据"的比例 |
| 可信度 | hall_rate | 答对且搜索的样本中"靠记忆"的比例 |
| 成本 | search/q | 每个问题平均搜索次数 |
| 成本 | oversearch_rate | "搜了但不必要搜"的比例 |
| 成本 | nosearch_rate | "没搜"的比例 |

评测时 trust 统一使用**反事实注入**（不管训练用什么信号），保证跨实验可比。

---

## 运行环境

- 集群：SLURM 管理，L40S (48GB) / RTX 4090 (24GB) / A100 (80GB)
- 框架：veRL (GRPO/PPO)，基于 Ray 分布式
- Retriever：基于 Pyserini BM25（Wikipedia index）
- Judge：Qwen2.5-3B-Instruct（FastAPI 服务）
