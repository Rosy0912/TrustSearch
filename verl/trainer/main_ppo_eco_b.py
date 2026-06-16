# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""EcoSearch training entry: Economic Reward Design for self-evolving search agents.

Reward = risk-adjusted net profit. ONLY R(tau) differs from Search-R1 baseline;
GRPO objective and retrieved-token masking are unchanged, so any EM/cost/trust
change is attributable to the reward.

Per-rollout economic dimensions
  - revenue  R_perf  = EM in {0,1}
  - trust    D_trust = groundedness of the answer in retrieved docs (lite: binary
               substring grounding with a small floor for correct-but-ungrounded).
               NER/entity-level coverage is left as an ablation (see paper) to
               avoid per-batch NER cost & noise.
  - cost     C       = N_search / N_budget(t)   (budget is self-evolving, sec 2.4)

Reward (completed, fixes the multiply-only gradient-sparsity flaw of the raw
formula by giving *wrong* rollouts a cost-graded signal so GRPO groups are not
all-zero early in training):

  answer is None                          -> 0
  no-search & correct                     -> 1 + bonus            (zero-cost profit)
  no-search & wrong                       -> -penalty             (overconfident loss)
  search    & wrong                       -> -lambda_cost * C     (cost-graded, keeps gradient)
  search    & correct                     -> R_perf * D_trust / (1 + C)

Self-evolving budget: every M train reward-calls, EMA(EM) & EMA(trust) shrink the
budget when the agent is strong & grounded (push it to do more with fewer
searches) and relax it when the agent is weak. Closed loop:
  agent stronger -> budget down -> cost up -> same behavior worth less -> agent
  forced to be more precise / search less -> stronger ...

Every call logs the 3 economic dimensions (EM / trust / search / no-search rate /
budget / reward) so performance, trust and cost can all be tracked from the log.
Validation reward = plain EM (comparable to baseline) but still logs trust/cost.
"""

from verl import DataProto
import torch
from verl.utils.reward_score import qa_em
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
import re
import numpy as np


_SEARCH_RE = re.compile(r"<search>", re.DOTALL)


class EcoRewardManager():
    """EcoSearch economic reward manager.

    mode='eco': economic risk-adjusted net-profit reward (train).
    mode='em' : plain EM reward (val), still logs trust/cost for multi-dim eval.
    """

    # --- economic reward params ---
    # Design principles (calibrated against real rollout stats: search/q≈2.0, EM≈0.25):
    #   - budget=2 => C=n_search/2, so C≈1 for typical rollout => 1/(1+C)≈0.5 (same scale as EM reward)
    #   - LAMBDA_COST=0: NO negative reward early in training; wrong rollouts get 0.
    #     Cost manifests ONLY through the denominator on correct rollouts (乘性惩罚),
    #     so reward scale stays positive and compatible with GRPO group baseline.
    #     Negative signals reintroduced later via self-evolving budget tightening.
    #   - UNGROUNDED_FLOOR=0.4: correct-but-ungrounded gets 0.4/(1+C)≈0.2 vs grounded 1/(1+C)≈0.5
    #     Discriminates PARAM_HALL from TRUE_TOOL without collapsing to near-zero.
    #   - BONUS=0.2, PENALTY=0.1: small perturbations; PENALTY intentionally tiny so no-search
    #     wrong doesn't dominate before no-search rate rises.
    BONUS = 0.2
    PENALTY = 0.1
    LAMBDA_COST = 0.0
    # Defaults; overridden in __init__ by env vars ECO_FLOOR / ECO_BUDGET_INIT
    UNGROUNDED_FLOOR = 0.4

    # --- self-evolving budget ---
    N_MAX = 4
    N_MIN = 1
    N_BUDGET_INIT = 2
    M_PERIOD = 25
    THETA_HIGH = 0.40
    THETA_LOW = 0.25
    THETA_D = 0.50
    EMA_M = 0.9

    def __init__(self, tokenizer, num_examine, format_score=0., reward_mode='eco') -> None:
        import os
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.format_score = format_score
        self.reward_mode = reward_mode
        # Ablation overrides via environment variables
        self.ungrounded_floor = float(os.environ.get("ECO_FLOOR", str(self.UNGROUNDED_FLOOR)))
        self.n_budget = float(int(os.environ.get("ECO_BUDGET_INIT", str(self.N_BUDGET_INIT))))
        print(f"[ECO-INIT] floor={self.ungrounded_floor} budget_init={self.n_budget} "
              f"lambda={self.LAMBDA_COST} bonus={self.BONUS} penalty={self.PENALTY}")
        self.ema_em = 0.0
        self.ema_trust = 0.0
        self._init = False
        self._calls = 0

    def _sample_stats(self, solution_str, ground_truth):
        answer = qa_em.extract_solution(solution_str=solution_str)
        n_search = len(_SEARCH_RE.findall(solution_str))
        if answer is None:
            return {'answer': None, 'correct': False, 'grounded': False, 'n_search': n_search}
        correct = bool(qa_em.em_check(answer, ground_truth['target']))
        ans_norm = qa_em.normalize_answer(answer)
        info_norm = qa_em.extract_information_blocks(solution_str)
        grounded = bool(ans_norm) and (ans_norm in info_norm)
        return {'answer': answer, 'correct': correct, 'grounded': grounded, 'n_search': n_search}

    def _eco_score(self, st):
        if st['answer'] is None:
            return 0.0
        n_search = st['n_search']
        # no-search boundary (self-knowledge)
        if n_search == 0:
            return (1.0 + self.BONUS) if st['correct'] else (-self.PENALTY)
        C = n_search / max(1e-6, self.n_budget)
        if not st['correct']:
            # gradient-keeping cost signal on wrong-with-search rollouts
            return -self.LAMBDA_COST * C
        # correct & searched: risk-adjusted net profit
        d_trust = 1.0 if st['grounded'] else self.ungrounded_floor
        return d_trust / (1.0 + C)

    def __call__(self, data: DataProto):
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        already_print_data_sources = {}

        n_total = 0
        n_correct = 0
        sum_trust = 0.0
        sum_search = 0.0
        n_nosearch = 0
        sum_reward = 0.0

        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)
            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch['data_source']

            st = self._sample_stats(sequences_str, ground_truth)

            if self.reward_mode == 'em':
                score = 1.0 if (st['answer'] is not None and st['correct']) else self.format_score
            else:
                score = self._eco_score(st)

            reward_tensor[i, valid_response_length - 1] = score

            # multi-dimensional bookkeeping
            n_total += 1
            sum_reward += score
            sum_search += st['n_search']
            if st['n_search'] == 0:
                n_nosearch += 1
            if st['correct']:
                n_correct += 1
                sum_trust += (1.0 if st['grounded'] else 0.0)

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        if n_total > 0:
            em = n_correct / n_total
            trust = (sum_trust / n_correct) if n_correct > 0 else 0.0   # grounding rate among correct
            search_mean = sum_search / n_total
            nosearch_rate = n_nosearch / n_total
            reward_mean = sum_reward / n_total
            tag = 'ECO-VAL' if self.reward_mode == 'em' else 'ECO'
            if self.reward_mode == 'eco':
                # EMA + self-evolving budget update
                if not self._init:
                    self.ema_em, self.ema_trust = em, trust
                    self._init = True
                else:
                    m = self.EMA_M
                    self.ema_em = m * self.ema_em + (1 - m) * em
                    self.ema_trust = m * self.ema_trust + (1 - m) * trust
                self._calls += 1
                if self._calls % self.M_PERIOD == 0:
                    old = self.n_budget
                    if self.ema_em > self.THETA_HIGH and self.ema_trust > self.THETA_D:
                        self.n_budget = max(self.N_MIN, self.n_budget - 1)
                    elif self.ema_em < self.THETA_LOW:
                        self.n_budget = min(self.N_MAX, self.n_budget + 1)
                    if self.n_budget != old:
                        print(f"[ECO-BUDGET] call={self._calls} ema_em={self.ema_em:.3f} "
                              f"ema_trust={self.ema_trust:.3f} budget {old:.0f}->{self.n_budget:.0f}")
            print(f"[{tag}] call~{self._calls} | PERF em={em:.3f} | TRUST grnd@correct={trust:.3f} "
                  f"| COST search/q={search_mean:.3f} nosearch={nosearch_rate:.3f} budget={self.n_budget:.0f} "
                  f"| reward={reward_mean:.3f}")

        return reward_tensor


import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})
    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup
    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup
    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }
    global_pool_id = 'global_pool'
    resource_pool_spec = {global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes}
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = EcoRewardManager(tokenizer=tokenizer, num_examine=0, reward_mode='eco')
    val_reward_fn = EcoRewardManager(tokenizer=tokenizer, num_examine=1, reward_mode='em')

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn,
                            )
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
