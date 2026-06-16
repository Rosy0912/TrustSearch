# TrustSearch 过夜运行交接 (2026-06-15 21:53 起)

## 一句话
4 组消融 + co-evolution + 3 探针服务全部健康运行，明早可看三维趋势 + 2 轮共演化边界迁移。

## 正在跑的作业（共 8 个核心 + 支撑服务 + 76 dump 分片）

| 作业 | 节点 | 显存 | 奖励配置 | 验证目的 | 状态 |
|---|---|---|---|---|---|
| ts_full | A100 node1 | 80G | EM + gate(1-bound)·flip − cost | 完整方法 | ✅ step10 |
| ts_nogate | ADA6000 node9 | 48G | EM + flip − cost（无门控） | 门控必要性(防过度搜索) | ✅ 刚过OOM(step1) |
| ts_nocost | L40S node10 | 48G | EM + gate·flip（无成本项） | 成本项作用 | ✅ step8 |
| ts_baseline | L40S node11 | 48G | EM only（=Search-R1） | 参照基线 | ✅ step9 |
| psrv_full/nogate/nocost | 4090 node7 | - | 探针热替换服务(8002/3/4) | 给probe奖励 | ✅ |
| coevolve | 4090 node6 | - | 监测checkpoint→refit探针→热替换 | self-evolution | ✅ 监测中 |
| cf_judge / cf_judge_b | node9/6 | - | 独立反事实判定(评测用) | 三维"可信"测量 | ✅ |
| retriever ×4 | node4/5/7/11 | - | 检索服务 | - | ✅ |
| extra-dump ×76 | 3090/4090 | - | 扩充探针数据池(dumps/base_extra) | 占空卡/数据量曲线 | 🟢 跑着 |

## 时间估算（关键）
- **每步 ~12-13 分钟**（gen 占大头，~400s/步）
- 当前 step 8-10，明早 ~10h 后 → **各实验约 step 55-65**
- ⚠️ total_training_steps=400 跑完需 ~80h，**明早不会跑完**，但消融趋势 55-65 步足够看出方向差异
- **co-evolution refit**：save_freq=25 → 约凌晨1点(step25)、早6点(step50) 各触发一次 → 明早有 **2 个边界迁移数据点**

## 明早验收清单
```bash
cd /home/jmzhang/self-evolution/Search-R1
squeue -u $USER | grep -E 'ts_|psrv|coevol'        # 1. 作业是否都还活着
# 2. 三维趋势（每组对比）：
for j in full nogate nocost baseline; do echo "=== $j ==="; \
  grep -aE 'val/perf/em|val/trust/hall_rate|val/cost/search_per_query' logs/ts_${j}_*.out | tail -6; done
cat logs/boundary_migration.csv                     # 3. 共演化边界扩张（核心self-evolution证据）
ls verl_checkpoints/ts_*/                            # 4. 各实验checkpoint
```

## 预期结论方向（待验证）
- **full vs baseline**：trust↑(hall_rate↓)、EM 不降、cost↓ → 三维同向
- **full vs nogate**：nogate 应在已知题过度搜索(search/q↑) → 证明门控必要
- **full vs nocost**：nocost 的 search/q 更高 → 证明成本项作用
- **边界迁移图**：闭卷正确率逐轮上升 = 知识边界扩张 = self-evolution 硬证据

## 已知风险/缺口
- 仍卡代理：PopQA/TriviaQA 下不了 → E0 标尺、OOD 迁移、完整7集评测待恢复（明早若代理通可补）
- 探针漂移：本轮靠 detached 奖励 + co-evolution refit 兜底（已接线）
- 若某组中途崩：full/nocost/baseline 三组 + 共演化已足够支撑核心三维结论
