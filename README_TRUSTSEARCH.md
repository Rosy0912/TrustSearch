# Search-R1 + TrustSearch

This repository contains **Search-R1** (an RL framework for training reasoning-and-searching
interleaved LLMs, built on [veRL](https://github.com/volcengine/verl)) together with
**TrustSearch**, our extension that rewards *trustworthy and cost-efficient* tool use.

- Original Search-R1 usage and theory: see [`README.md`](README.md).
- This file (`README_TRUSTSEARCH.md`) is the **hand-off guide** for reproducing the
  baseline vs. TrustSearch comparison on your own cluster.

---

## 1. What is TrustSearch?

Search-R1 trains the model with a plain Exact-Match (EM) reward: `1` if the final
answer is correct, else `0`. It cannot tell *whether the model actually used the
retrieved documents* or just answered from parametric memory, and it does not penalize
unnecessary searches.

TrustSearch optimizes three things at once — **performance, trust, and cost**:

```
answer = None              -> 0
correct & no_search        -> 1 + BONUS            (knew the answer, no need to search)
wrong   & no_search        -> -PENALTY             (should have searched)
wrong   & searched         -> 0
correct & searched         -> D_trust * (1 - alpha * norm_cost)
```

`D_trust` is an **online counterfactual** signal: for every correct & searched rollout,
we corrupt the retrieved docs (replace the gold answer with a fake entity) and let the
model answer again.

| Counterfactual outcome | meaning | D_trust |
|---|---|---|
| answer **changes** under fake docs | really used the tool (`TRUE_TOOL`) | `1.0` |
| still correct on fake docs | ignored the tool / parametric (`PARAM_HALL`) | `0.3` (floor) |
| ambiguous | – | `0.6` |

`norm_cost = max(0, n_search - 1) / (MAX_SEARCH - 1)` so the **first search is free** and
extra searches are penalized by `alpha` (default `0.5`, `MAX_SEARCH=2`).

Key source files:

| File | Role |
|---|---|
| `verl/trainer/main_ppo_eco.py` | TrustSearch reward manager (online counterfactual) + 3-dim validation |
| `cf_judge_server.py` | HTTP service that performs the counterfactual judging |
| `verl/trainer/main_ppo.py` | Search-R1 baseline (plain EM) + 3-dim validation logging |
| `search_r1/llm_agent/generation.py` | rollout / search loop (with retriever retry) |
| `search_r1/search/retrieval_server.py` | local dense retriever (FastAPI + e5 + FAISS) |

---

## 2. Environment

```bash
conda create -n searchr1 python=3.9 -y
conda activate searchr1

# veRL + Search-R1 deps (see original README.md for the full list)
pip install -r requirements.txt
pip install -e .                      # installs the local `verl` package

# vLLM 0.5.4 backend is expected; flash-attn optional (see install_flash_attn.sbatch)
```

The counterfactual judge (`cf_judge_server.py`) additionally needs:
`fastapi`, `uvicorn`, `transformers`, `torch` (already covered by the env above).

---

## 3. Download data & models

```bash
bash download_all.sh          # or: sbatch download.sbatch
```

This fetches:

- **Corpus**: `data/retriever_data/wiki-18.jsonl` (~21M passages)
- **FAISS index**: `data/retriever_data/e5_Flat.index`
- **QA data**: `data/nq_search/{train,test}.parquet`

You also need two HuggingFace models (set your own local paths):

- Policy / base model: `Qwen/Qwen2.5-3B-Instruct`
- Retriever encoder: `intfloat/e5-base-v2`

> NOTE: corpus, index, checkpoints and parquet data are **git-ignored** (too large for
> GitHub). Every user must download them locally.

---

## 4. Cluster-specific settings (IMPORTANT — edit before running)

All `*.sbatch` files were written for our SLURM cluster and **hardcode** node names,
partitions and service IPs. You must change them for your machine:

- `#SBATCH --partition=...`, `#SBATCH --nodelist=...`, `#SBATCH --gres=gpu:N`
- `RETRIEVER_URL="http://<retriever-node-ip>:8000/retrieve"`
- `CF_JUDGE_URL="http://<cf-judge-node-ip>:8001/judge_batch"`
- model paths: `actor_rollout_ref.model.path=...` (in `*_common.sh`) and
  `--retriever_model` (in `start_retriever*.sbatch`)

If you run everything on a single multi-GPU node, just point all URLs at `127.0.0.1`.

---

## 5. Run order

The trainers need the **retriever** (and, for TrustSearch, the **cf_judge**) up first.

### Step 1 — start the retriever(s)

```bash
sbatch start_retriever.sbatch       # serves http://<node>:8000/retrieve
```

Wait until it logs that the index is loaded (loading 21M passages takes several minutes).
You can give the baseline and TrustSearch their own retriever to avoid contention
(`start_retriever_b.sbatch`, `start_retriever_c.sbatch`).

### Step 2 — start the counterfactual judge (TrustSearch only)

```bash
sbatch start_cf_judge.sbatch        # serves http://<node>:8001/judge_batch
```

### Step 3 — launch training

```bash
# Search-R1 baseline (plain EM reward)
sbatch train_baseline.sbatch

# TrustSearch (online counterfactual trust + cost)
sbatch train_eco_full.sbatch
```

Both use identical hyper-parameters (GRPO, Qwen2.5-3B-Instruct, n_agent=5,
max_turns=2); the **only difference is the reward**, so the comparison is clean.

### Step 4 — offline 3-dimension evaluation of a checkpoint

```bash
# edit MODEL_PATH inside, then:
sbatch eval_ckpt50.sbatch           # eval a trained checkpoint
sbatch eval_base.sbatch             # eval the untrained base model
```

---

## 6. Reading the metrics

Validation runs every `test_freq` steps and reports all three dimensions
(grep the training log for `val/` or the `[TRUST-VAL]` / `[BASE-VAL]` lines):

| Dimension | Metric | Meaning |
|---|---|---|
| Performance | `val/perf/em` (= `val/test_score/nq`) | Exact-Match accuracy |
| Trust | `val/trust/trust_at_correct` | avg counterfactual trust of correct answers |
| Trust | `val/trust/hall_rate` | fraction of correct answers that look parametric (`PARAM_HALL`) |
| Cost | `val/cost/search_per_query` | avg searches per question |
| Cost | `val/cost/nosearch_rate` | fraction answered with no search |
| Cost | `val/cost/oversearch_rate` | fraction with >1 search |

During training the reward manager also prints per-batch stats:

```
[TRUST] call~N | PERF em=... | TRUST trust@correct=... cf[TRUE=.. HALL=.. AMB=..] | COST search/q=.. nosearch=.. alpha=.. | reward=..
```

---

## 7. Reward knobs (env vars for TrustSearch)

Set in `train_eco_full.sbatch`:

| Var | Default | Meaning |
|---|---|---|
| `ECO_USE_CF` | `1` | use online counterfactual D_trust (`0` = temporal proxy) |
| `ECO_FLOOR` | `0.3` | D_trust for `PARAM_HALL` |
| `ECO_ALPHA` | `0.5` | search-cost strength |
| `CF_JUDGE_URL` | – | counterfactual judge endpoint |
| `ECO_COST_ONLY` / `ECO_TRUST_ONLY` / `ECO_NO_SELF` | `0` | ablations |

---

## 8. Layout

```
verl/                      # veRL core (modified: trainers, fsdp workers, reward)
  trainer/main_ppo.py      #   Search-R1 baseline entry (+3-dim val)
  trainer/main_ppo_eco.py  #   TrustSearch entry (online counterfactual)
search_r1/                 # retriever server + rollout/search agent
cf_judge_server.py         # counterfactual judge HTTP service
train_baseline.sbatch      # baseline launcher          + baseline_train_common.sh
train_eco_full.sbatch      # TrustSearch launcher        + eco_train_common.sh
start_retriever*.sbatch    # retriever services
start_cf_judge.sbatch      # cf judge service
eval_*.sbatch              # offline 3-dim evaluation    + eval_3dim_common.sh
data/  verl_checkpoints/   # (git-ignored) download / produced locally
```

Built on Search-R1 (Jin et al.) and veRL. See `LICENSE`.
