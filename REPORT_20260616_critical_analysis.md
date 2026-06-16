# Critical Analysis: Self-Evolution Narrative Stress Test
**Date**: 2026-06-16  
**Status**: 🔴 Narrative under threat, awaiting ts_static experiment

---

## Executive Summary

The CKA + frozen-probe analysis revealed three findings that **threaten the "co-evolution is necessary" narrative**:
1. ts_full frozen probe AUC **improves** (0.82→0.89), contradicting "drift causes probe failure"
2. CKA is a red herring (same drift magnitude → opposite probe fates)  
3. ts_nogate self-probe AUC=1.0 is a **model collapse artifact**, not genuine capability

However, deep calibration analysis reveals a **subtle but potentially decisive mechanism** through which co-evolution could still matter: **reward signal fidelity** (Brier score, FN rate), not classification accuracy (AUC).

---

## 1. ts_nogate: Confirmed Model Collapse 🚨

**Verdict**: ts_nogate is NOT "representation drift without gate" — it is **complete behavioral collapse**.

| Metric | r25 | r50 | r75 |
|--------|-----|-----|-----|
| EM rate | 0.025 | 0.019 | **0.000** |
| Flip rate (y_flip=1) | 86.4% | 96.9% | **100.0%** |
| Search/query | 2.04 | 1.05 | 1.03 |
| Unique answers | 95.5% | 86.6% | **62.5%** |
| n_neg (y_flip=0) | 197 | 43 | **0** |

**What happened**: Without the gate, the model progressively stopped learning useful answers (EM→0) while its flip rate went to 100% (trivially — if you never answer correctly, every search "flips" the outcome from wrong to wrong). The model collapsed into verbose non-answers.

**Implication for frozen probe AUC=0.5**: This is NOT "probe failing to read a drifted representation." It's simply `len(np.unique(y)) < 2` → the code returns 0.5 by default because AUC is undefined with a single class. **This data point cannot be used to argue "probe degradation without gate."**

**Correct interpretation**: The gate prevents **behavioral collapse** (reward hacking via degenerate outputs), not "representation drift." The story should be: "Gate prevents mode collapse in the trust reward."

---

## 2. Frozen Probe AUC Trend: Robust, Not an Artifact

### Setup resolved
- **V-shape** (0.995→0.89→0.96): Using r25 co-evolved probe evaluated on later rounds. This is in-sample→OOD→partial-recovery. Not meaningful for "r0 probe degradation" narrative.
- **Monotone rise** (0.82→0.84→0.89): Using r0 (base250b) probe. This IS the relevant setup for ts_static.

### Balanced sampling control
Even with matched class sizes (n_neg=n_pos=100), AUC rises: **0.822 → 0.839 → 0.887**. Not a prevalence artifact.

### Mechanism
The `separation` (mean pred for y=1 minus mean pred for y=0) increases: **0.451 → 0.494 → 0.588**. The training is **sharpening** the cognitive direction that the r0 probe reads. Gate preserves this direction while the model learns.

### Locked setup for paper
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Probe | `probes/base250b/commit_probe.pkl` | The actual probe that ts_static would use |
| Position/Layer | Xa_21 | Same as probe training |
| Anchor | r0 (base model) | Meaningful for ts_static comparison |
| Evaluation data | coevo dumps, NOT base dumps | Same query set across rounds |
| AUC computation | Standard, no gid averaging | Per-sample prediction |

---

## 3. CKA: Demoted from "Main Evidence" to "Background Context"

**Why CKA fails as the main argument:**
- ts_full CKA(L24) r25→r75: **0.76** → frozen probe AUC **improves** to 0.89
- ts_nogate CKA(L24) r25→r75: **0.72** → frozen probe **undefined** (model collapsed)
- Same CKA magnitude, completely different outcomes → CKA cannot predict probe fate

**What CKA CAN say (limited):**
- Representation geometry does shift during RL (CKA < 1.0)
- The shift is similar in magnitude regardless of whether the model works or crashes
- Therefore, geometric drift is **necessary but not sufficient** for failure

**Revised role in paper**: One sentence in background. Not a figure. Not part of the causal argument.

---

## 4. The True Argument for Co-Evolution: Reward Fidelity (Not AUC)

### Key insight: AUC ≠ reward quality

The frozen r0 probe has decent AUC (0.82-0.89), but **terrible calibration**:

| Round | Brier (frozen) | Brier (co-evolved) | FN% (frozen) | FN% (co-evolved) |
|-------|---------------|--------------------|--------------|--------------------|
| r25 | 0.238 | **0.019** | 31.6% | **1.2%** |
| r50 | 0.263 | **0.057** | 34.4% | **0.9%** |
| r75 | 0.155 | **0.011** | 18.6% | **0.5%** |

**Translation**: The frozen probe **misses 30% of beneficial searches** (gives them zero reward). The co-evolved probe misses only 1%. This 30× difference in false-negative rate means:

1. **Under-rewarding beneficial search**: Policy receives noisy, inconsistent signal about which searches helped
2. **Reward sparsity accumulation**: Over 100+ steps, systematically missing 30% of positive signals could slow learning or shift the exploration-exploitation balance
3. **Calibration drift**: pred<0.2 bucket has actual_pos_rate=0.57-0.73 → probe's "confident negative" is wrong majority of the time

### Why this MIGHT not matter (→ ts_static will tell):
- GRPO uses **advantages** (relative within batch), not absolute reward values
- If all flip-reward samples are equally under-predicted, the rank ordering is preserved
- AUC measures rank ordering → if AUC is high (0.89), advantages might be fine

### Why this MIGHT matter (→ ts_static will tell):
- The gate mechanism uses `probe_score * (1 - boundary)` → absolute scale matters
- 30% FN means some samples get gate=0 reward when they should get gate=0.8
- Over many steps, this biases toward "don't search when unsure" → conservative policy

---

## 5. ts_static Experiment Design (Ready to Run)

### Files created:
- `ts_static_grpo.sbatch` — Training script (identical to ts_full except probe server)
- `probe_server_static.py` — Probe server with frozen r0 heads + live encoder
- `ts_static_probe_server.sbatch` — Server launch script  
- `ts_static_encoder_watcher.py` — Only updates encoder checkpoint, never probes
- `ts_static_manifest.json` — Initial manifest

### Key design decisions:
- **Encoder**: Tracks current policy (same as ts_full) — isolates probe head contribution
- **Probe heads**: Frozen at `probes/base250b/{commit,boundary}_probe.pkl`
- **Everything else**: Identical to ts_full (same hyperparams, gate, cost term)

### How to run:
```bash
# Node 1 (GPU server for probe):
sbatch ts_static_probe_server.sbatch

# Same node or another (background, no GPU):
python ts_static_encoder_watcher.py --ckpt_root verl_checkpoints/ts_static/actor &

# Node 2 (4-GPU training):
VARIANT=full sbatch ts_static_grpo.sbatch  # but with PROBE_URL pointing to static server
```

### Success criteria:
| Outcome | Meaning | Action |
|---------|---------|--------|
| ts_static EM ≈ ts_full EM (within 2pp) | Co-evolution not necessary | Downgrade self-evolution to "engineering robustness" |
| ts_static EM < ts_full EM by ≥5pp | Co-evolution matters | Paper narrative: "AUC masks calibration drift that accumulates as reward bias" |
| ts_static collapses (EM→0) | Co-evolution prevents mode collapse | Strongest result: frozen probe's FN bias eventually destabilizes learning |

---

## 6. Revised Narrative Options (Post ts_static)

### If co-evolution IS necessary:
> "While a frozen probe maintains high AUC (0.89), its calibration degrades — 30% of beneficial searches receive zero trust-reward (FN). Over training, this systematic under-rewarding accumulates into X% EM loss (Table N). Co-evolution maintains <1% FN through continuous probe refit, delivering cleaner reward signal that sustains balanced exploration."

Key: The story is about **reward signal fidelity**, not about "the probe breaks." This is more subtle, more defensible, and doesn't contradict the AUC data.

### If co-evolution is NOT necessary:
> "Our analysis reveals that the gate mechanism is the critical component: it prevents behavioral collapse (ts_nogate) by modulating trust-reward with the knowledge boundary. The probe itself is robust to representation drift (frozen probe AUC 0.82→0.89), and co-evolution provides marginal improvement in reward calibration (Brier 0.24→0.02) without significant impact on downstream performance."

Key: Gate is the hero, co-evolution is nice-to-have. Paper title might need adjustment.

---

## 7. Immediate Action Items

| Priority | Task | Status |
|----------|------|--------|
| 🔴🔴 | Run ts_static experiment | ⏳ Ready (sbatch files created) |
| 🔴 | ts_nogate narrative: rewrite as "mode collapse" not "drift" | ✅ Analysis complete |
| 🔴 | Lock analysis setup (r0 probe, Xa_21, per-sample AUC) | ✅ Decided |
| 🟠 | Remove CKA from main figures | ✅ Decision made |
| 🟡 | Add calibration (Brier/FN%) as new metric alongside AUC | Pending ts_static result |

---

## Appendix: Layer Drift Note

> "低层比高层漂移快、因为 RL 改 policy head 附近" — 这个解释是反的

Correct observation: L21 CKA drops more than L27. But "policy head nearby" would predict L27 (layer 35) changing most. The actual mechanism is likely:
- Early layers capture more input-dependent features (sensitive to behavior distribution shift)
- Later layers are more abstracted (stable under distribution shift)
- This is a generic property of deep networks, not specific to RL

**Action**: Describe the phenomenon. Do NOT provide mechanistic explanation unless we have ablation evidence.
