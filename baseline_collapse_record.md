# Baseline (纯 EM, 无 cost 约束) 三维指标记录 + 过度搜索坍塌证据

实验: `ts_grpo` job 73083 (node9), Qwen2.5-3B-Instruct, GRPO, max_turns=4
记录时间: 2026-06-22 21:11
终止原因: 过度搜索 hack 持续恶化 (search/q 单调逼近 3.0 上限), 已取得充分的对照证据。

## 各 checkpoint 验证集三维 (200 样本 inline val)

| step | EM (Performance) | trust@correct (Trust) | hall_rate | search/q (Cost) | oversearch |
|-----:|:----------------:|:---------------------:|:---------:|:---------------:|:----------:|
| 25   | 0.328 | 0.986 | 0.000 | 2.094 | 0.984 |
| 50   | 0.422 | 0.989 | 0.000 | 2.281 | 1.000 |
| 75   | 0.453 | 0.976 | 0.034 | 2.422 | 1.000 |

## 全量评测 (3610 样本) — baseline step50

| 指标 | 值 |
|------|-----|
| EM | 0.3518 |
| trust@correct | 0.9359 |
| hall_rate | 0.0691 |
| search/q | 2.2408 |
| oversearch_rate | 0.9788 |

(全量 EM 0.352 < inline val 0.422, 说明 inline 200 样本偏乐观; 全量更可靠)

## 过度搜索坍塌轨迹 (训练 train EM / search/q)

| call | train EM | search/q | oversearch |
|-----:|:--------:|:--------:|:----------:|
| 84 | 0.328 | 2.603 | 0.875 |
| 85 | 0.394 | 2.541 | 0.955 |
| 86 | 0.244 | 2.981 | 0.842 |
| 87 | 0.287 | 2.842 | 0.922 |

## 结论
- search/q 从 step50 的 2.28 单调恶化到 step87 的 ~2.8-3.0 (逼近 max_turns 上限)
- oversearch_rate 稳定在 ~95-100%: 几乎每个 query 都过度搜索
- 这是无 cost 约束的 GRPO 多轮 agent 通病 (over-search hack)
- 对照价值: 证明 IRIS 的 cost 约束维度是必需的; 但 cost 过重 (F2 cost=0.2) 会引发反向的 under-search 坍塌, 需 cost=0.05 的温和配方 (F10/F11) 找平衡点。
