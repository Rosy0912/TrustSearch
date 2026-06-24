# 实验交接报告 — 2026-06-22 晚

## 一、两个新 IRIS 实验（突出探针作用）— 均已成功训练起来

目标: 让探针(probe)信号在 reward 中更主导, 解决"提升太小"
原因: base=1.0 常数淹没探针项, GRPO 组内归一化后探针区分度低

| 实验 | Job | 节点 | 探针强化方式 | 关键参数 |
|------|-----|------|-------------|----------|
| A=ts_F13_probeA | 74369 | node9 4xADA6000 | 纯加性强化:grounding翻倍 | BAL_GROUND=0.8 NOSEARCH_BONUS=0.8 WRONG_GROUND_SCALE=0.3 W_COST=0.05 |
| B=ts_F13_probeB | 74361 | node9 4xADA6000 | 探针进base:grounded答对>蒙对 | PROBE_BASE=0.3 BAL_GROUND=0.5 WRONG_GROUND_SCALE=0.3 W_COST=0.05 |

共同点: 温和cost=0.05(避免F2 under-search崩塌), wrong_ground_scale=0.3(全错组也靠探针区分提供梯度)

### 代码改动(已隔离不影响他人)
- main_ppo_eco.py 新增 ECO_PROBE_BASE 开关(默认0=原行为). 开启时 correct base=(1-pb)+pb*flip
- ts_recipe.sbatch/ts_grpo.sbatch: 参数化PPO_MICRO_BS, 修复CF_JUDGE_URL默认值(node9->node1)

### 启动状态
- A(74369): step1通过backward, call~1 em=0.222, cf正常, 0 OOM
- B(74361): step9, em=0.194 search/q=2.083, 健康
- 两者跑到step250, save_freq/test_freq=25

## 二、全量评测结果(3610样本) baseline vs IRIS(F2 step50)

| 维度 | 指标 | Baseline | IRIS(ours) | 优劣 |
|------|------|:---:|:---:|:---:|
| Performance | EM | 0.3518 | 0.3569 | 持平略升 |
| Trust | trust@correct | 0.9359 | 0.9322 | 持平 |
| | hall_rate | 0.0691 | 0.0613 | 优 -11% |
| Cost | search/q | 2.2408 | 2.1638 | 优 |
| | oversearch | 0.9788 | 0.9387 | 优 |

结论: IRIS在Trust(幻觉)和Cost两维更好, Performance持平. 提升偏小->正是A/B要改进的.

## 三、关键教训(资源/配置)
1. RTX4090(24GB)完全不能跑此3B GRPO - step1 backward必OOM(F10/F11死于此)
2. A100(40GB)2卡也不行 - backward峰值差~5GB. 必须>=4卡
3. ADA6000/L40S(48GB)4卡=已验证可行配置(F12/A/B均成功). 2卡48GB也OOM
4. cf_judge会被坏输入触发CUDA assert永久崩溃 - 已加固(token clamp+try/except降级), 重启在node1(192.168.102.11:8001)

## 四、已终止/死亡的实验
- baseline(73083): 过度搜索坍塌(search/q->3.0), 三维记录于baseline_collapse_record.md, 主动终止
- F2_balanced(73081): under-search崩塌(cost=0.2太重), 保留作对照
- F10/F11: RTX4090 OOM死亡, 科学目的已被A覆盖
- F12_best(74359): 弱探针(grounding=0.4), 被强探针A(0.8)覆盖, 主动终止腾node9给A

## 五、当前运行任务
| Job | 名称 | 节点 | 说明 |
|-----|------|------|------|
| 74369 | A=ts_F13_probeA | node9 | 强加性探针训练中 |
| 74361 | B=ts_F13_probeB | node9 | 探针进base训练中 |
| 74358 | cf_judge(加固) | node1 | 0个500健康 |
| 73081 | F2(原已崩) | node8 | 对照保留 |
| 73080/73082/72568 | probe_srv/coevolve/retriever | - | 支撑服务正常 |

## 六、明早建议
1. 看A/B的step25/50验证三维: grep "step:" logs/ts_grpo_74369.out 和 _74361.out, 比较hall_rate/search_q/EM是否比F2 step50更好
2. 若A/B到step50三维全面优于baseline, 对其checkpoint跑全量评测(改ts_eval_full.sbatch的CKPT/TAG)
3. 警惕over-fit探针: 若A(grounding=0.8)出现"表演grounded但EM崩", 说明探针权重过强, 回退到0.6
