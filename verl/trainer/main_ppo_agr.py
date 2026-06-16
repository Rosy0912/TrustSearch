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
"""AGR (Adaptive Grounded Reward) training entry.

Identical to main_ppo.py / main_ppo_behavcf.py EXCEPT the reward weights are
*dynamic*: instead of a fixed groundedness gate (behavcf uses a constant
ungrounded_score=0.3), AGR lets the gate strength + search incentive + redundancy
penalty be driven by the agent's OWN running behavior statistics, forming an
agent<->reward co-evolution loop:

  stage 0->1 (can't use tool):     low search_freq  -> search incentive HIGH  (encourage any retrieval)
  stage 1->2 (uses tool, ungrounded): low grounding_rate -> gate LENIENT (g high, ~plain EM, don't over-punish)
                                       grounding_rate rises -> gate TIGHTENS (g -> g_low) push genuine grounding
  stage 2->3 (grounded but redundant): high redundancy -> redundancy penalty ACTIVATES

All statistics are EMA-smoothed (slow, momentum=0.9) and every weight is clipped
to a bounded range, so the non-stationary reward stays controllable (no high-freq
oscillation). Val uses plain EM so the metric is directly comparable to baseline.

The ONLY variable vs the static behavcf run is "fixed gate (0.3)" -> "adaptive
gate driven by behavior", isolating the closed-loop contribution.
"""

from verl import DataProto
import torch
from verl.utils.reward_score import qa_em
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
import re
import numpy as np


_SEARCH_RE = re.compile(r"<search>", re.DOTALL)


class AGRRewardManager():
    """Adaptive Grounded Reward manager.

    Train mode ('agr'): per-sample reward computed under the CURRENT adaptive
    weights; after scoring, the batch behavior stats update the EMA which drives
    the NEXT step's weights (1-step lag, intentionally slow-moving).

    Val mode ('em'): plain EM, fair & comparable to baseline.
    """

    # --- adaptive weight ranges (bounded for stability) ---
    G_HIGH = 0.7      # lenient gate when agent can't ground yet
    G_LOW = 0.1       # strict gate once grounding ability is high
    RG_TARGET = 0.6   # grounding-rate at which gate reaches full strictness
    SF_MIN = 0.9      # target avg #search per sample; below this -> incentivize
    B_MAX = 0.10      # max search incentive
    RR_THRESH = 0.6   # redundancy (frac with >=2 searches) above which penalize
    P_MAX = 0.10      # max redundancy penalty
    EMA_M = 0.9       # EMA momentum (slow update -> stable)

    def __init__(self, tokenizer, num_examine, format_score=0., reward_mode='agr') -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.format_score = format_score
        self.reward_mode = reward_mode
        # EMA state of behavior statistics
        self.ema_rg = 0.0   # grounding rate among correct answers
        self.ema_sf = 0.0   # mean search frequency
        self.ema_rr = 0.0   # redundancy rate
        self._initialized = False
        self._step = 0

    def _current_weights(self):
        """Map EMA behavior stats -> bounded adaptive weights."""
        g = self.G_HIGH - (self.G_HIGH - self.G_LOW) * min(1.0, self.ema_rg / self.RG_TARGET)
        g = float(np.clip(g, self.G_LOW, self.G_HIGH))
        b_search = self.B_MAX * max(0.0, (self.SF_MIN - self.ema_sf) / self.SF_MIN)
        b_search = float(np.clip(b_search, 0.0, self.B_MAX))
        p_red = self.P_MAX * max(0.0, (self.ema_rr - self.RR_THRESH) / max(1e-6, 1.0 - self.RR_THRESH))
        p_red = float(np.clip(p_red, 0.0, self.P_MAX))
        return g, b_search, p_red

    def __call__(self, data: DataProto):
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        # Weights for THIS batch come from EMA accumulated so far (1-step lag).
        g, b_search, p_red = self._current_weights()

        already_print_data_sources = {}
        n_correct = 0
        n_grounded_correct = 0
        sum_search = 0.0
        n_redundant = 0
        n_total = 0

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

            if self.reward_mode == 'em':
                # validation: plain EM, comparable to baseline
                score = qa_em.compute_score_em(
                    solution_str=sequences_str, ground_truth=ground_truth,
                    format_score=self.format_score)
            else:
                score = self._agr_score(sequences_str, ground_truth, g, b_search, p_red)
                # collect behavior stats for EMA update
                stat = self._sample_stats(sequences_str, ground_truth)
                n_total += 1
                sum_search += stat['n_search']
                if stat['n_search'] >= 2:
                    n_redundant += 1
                if stat['correct']:
                    n_correct += 1
                    if stat['grounded']:
                        n_grounded_correct += 1

            reward_tensor[i, valid_response_length - 1] = score

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        # update EMA with this batch (train only) and log the co-evolution
        if self.reward_mode != 'em' and n_total > 0:
            batch_rg = (n_grounded_correct / n_correct) if n_correct > 0 else 0.0
            batch_sf = sum_search / n_total
            batch_rr = n_redundant / n_total
            if not self._initialized:
                self.ema_rg, self.ema_sf, self.ema_rr = batch_rg, batch_sf, batch_rr
                self._initialized = True
            else:
                m = self.EMA_M
                self.ema_rg = m * self.ema_rg + (1 - m) * batch_rg
                self.ema_sf = m * self.ema_sf + (1 - m) * batch_sf
                self.ema_rr = m * self.ema_rr + (1 - m) * batch_rr
            self._step += 1
            ng, nb, npr = self._current_weights()  # weights for NEXT step
            print(f"[AGR] step~{self._step} | batch(rg={batch_rg:.3f} sf={batch_sf:.3f} rr={batch_rr:.3f}) "
                  f"| ema(rg={self.ema_rg:.3f} sf={self.ema_sf:.3f} rr={self.ema_rr:.3f}) "
                  f"| weights_used(gate={g:.3f} search+={b_search:.3f} red-={p_red:.3f}) "
                  f"-> next(gate={ng:.3f} search+={nb:.3f} red-={npr:.3f})")

        return reward_tensor

    def _sample_stats(self, solution_str, ground_truth):
        answer = qa_em.extract_solution(solution_str=solution_str)
        n_search = len(_SEARCH_RE.findall(solution_str))
        if answer is None:
            return {'correct': False, 'grounded': False, 'n_search': n_search}
        correct = bool(qa_em.em_check(answer, ground_truth['target']))
        ans_norm = qa_em.normalize_answer(answer)
        info_norm = qa_em.extract_information_blocks(solution_str)
        grounded = bool(ans_norm) and (ans_norm in info_norm)
        return {'correct': correct, 'grounded': grounded, 'n_search': n_search}

    def _agr_score(self, solution_str, ground_truth, g, b_search, p_red):
        st = self._sample_stats(solution_str, ground_truth)
        answer = qa_em.extract_solution(solution_str=solution_str)
        if answer is None:
            return 0.0
        # adaptive groundedness gate
        if st['correct']:
            base = 1.0 if st['grounded'] else g
        else:
            base = self.format_score
        # stage 0->1: encourage using the tool at all (decays as search_freq rises)
        search_term = b_search if st['n_search'] >= 1 else 0.0
        # stage 2->3: discourage redundant retrieval (activates when redundancy high)
        red_term = -p_red if st['n_search'] >= 2 else 0.0
        score = base + search_term + red_term
        return float(np.clip(score, 0.0, 1.0))


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
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
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

    reward_fn = AGRRewardManager(tokenizer=tokenizer, num_examine=0, reward_mode='agr')
    val_reward_fn = AGRRewardManager(tokenizer=tokenizer, num_examine=1, reward_mode='em')

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
