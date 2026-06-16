# TrustSearch — Foundation Diagnostics + Probe (E0/E0.5/E1)

Model: `sr1-baseline-em/actor/global_step_250` (Qwen2.5-3B, 36 layers)
Data : NQ + HotpotQA train, **14,175 rollouts** (G=5)
Probe: group-aware split by question (no question leaks train->test). Model selection over
       position x label x model x layer.

## Two probes (final)
| probe | position | layer | label | model | AUC |
|---|---|---|---|---|---|
| **commit (flippability)** | Xa = answer-span last token | L21 | flip (empty `<information>`) | MLP(64) | **0.824** |
| **boundary (closed-book)** | Xp = prompt last token | L27 | closed-book correct | LR | 0.664 |

Saved: `probes/base250b/{commit,boundary}_probe.pkl`

### Model-selection findings (flip AUC, group-CV)
- **answer-span last token (Xa) >> "<answer>" tag token (Xc)**  (0.82 vs 0.77)
- **MLP > logistic regression**  (0.82 vs 0.76)
- flip (info-only) >= flip2 (also strip `<think>`); stripping think did NOT help -> kept info-only
- best layer L21 (58% depth); higher layers similar
- vs first pass (Xc/LR/L21 = 0.68) -> **+0.14 AUC** from position + MLP + more data

## E1 — flippability (KILL-OR-PROCEED): PASS
- commit probe predicts flippability **AUC=0.824** (honest, group-split).
- **82% of GRPO groups are zero-variance** (1981 all-wrong + 523 all-right / 3039).
- flip has within-group variance in **36.1% of all-wrong (dead-gradient) groups** -> dense
  signal rescues ~1/3 of otherwise-zero-gradient groups. Core mechanism real.

## E0 — boundary ruler: PARTIAL (needs PopQA/TriviaQA)
- boundary probe AUC ~0.66 (weaker; closed-book is harder / range compressed with only 2 sets).
- read tracks cb rate: hotpotqa read=0.278(cb 0.246) > nq read=0.204(cb 0.174).
- Full ruler PopQA->NQ->TriviaQA BLOCKED (proxy down).

## E0.5 — orthogonality (control EM): PASS (directional)
- within EM==1: hotpotqa read=0.522, nq=0.605 ; within EM==0: hotpotqa=0.789, nq=0.776
- read varies across datasets WITHIN a fixed EM bin -> probe carries info beyond EM.
- EM==0 rollouts FAR more flippable (~0.78) than EM==1 (~0.55) -> wrong answers are much
  more evidence-dependent / unstable. Strong, consistent signal.

## Verdict on GRPO-readiness
- flip probe AUC **0.82** is comfortably usable as a DENSE SHAPING signal (with EM anchor +
  boundary gating). Boundary probe (0.66) is weaker -> use as soft gate, improve with PopQA.

## Blocked / TODO
- proxy `192.168.102.101:7890` down -> can't fetch PopQA/TriviaQA/2Wiki/MuSiQue/Bamboogle
  (needed for E0 real ruler AND final 3-dim eval).
- next: probe service (external scalar -> detached reward) + main_ppo_eco ECO_TRUST_SIGNAL=probe
  (ground = gate(boundary) x flip) + E2 vs self-consistency baseline.
