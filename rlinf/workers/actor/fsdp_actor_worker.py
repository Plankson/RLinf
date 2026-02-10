# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from functools import partial
from typing import Optional
import torch.nn.functional as F
import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn
from torch.distributed.tensor import DTensor
from torch.multiprocessing.reductions import reduce_tensor

import rlinf.algorithms  # noqa: F401
from rlinf.algorithms.registry import calculate_adv_and_returns, policy_loss
from rlinf.algorithms.utils import (
    kl_penalty, compute_v_sg, asymmetric_l2_loss
)
from rlinf.config import SupportedModel
from rlinf.data.embodied_io_struct import Trajectory, convert_trajectories_to_batch
from rlinf.data.io_struct import BatchResizingIterator, RolloutResult
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import (
    FSDPModelManager,
)
from rlinf.models import get_model
from rlinf.scheduler import Channel, Cluster, CollectiveGroupOptions, Worker
from rlinf.utils.data_iter_utils import get_iterator_k_split
from rlinf.utils.distributed import all_reduce_dict, masked_normalization
from rlinf.utils.distributed import (
    compute_rollout_metrics as compute_math_rollout_metrics,
)
from rlinf.utils.metric_utils import (
    append_to_dict,
    compute_loss_mask,
    compute_rollout_metrics,
    compute_split_num,
)
from rlinf.utils.nested_dict_process import (
    put_tensor_device,
    split_dict_to_chunk,
)
from rlinf.utils.placement import (
    HybridComponentPlacement,
    ModelParallelComponentPlacement,
)
from rlinf.utils.utils import (
    clear_memory,
    compute_entropy_from_logits,
    compute_logprobs_from_logits,
    cpu_weight_swap,
    get_loss_agg_func,
    masked_mean,
    reshape_entropy,
    retrieve_model_state_dict_in_cpu,
)
from rlinf.workers.rollout.utils import RankMapper


def process_nested_dict_for_adv(nested_dict, rollout_epoch):
    """
    original shape: [rollout_epoch x n_chunk_steps, bsz, num_action_chunks, ...]
    target shape: [n_chunk_steps, rollout_epoch x bsz, num_action_chunks, ...]
    """
    ret_dict = {}
    for key, value in nested_dict.items():
        if isinstance(value, torch.Tensor):
            new_value = value.reshape(
                rollout_epoch, -1, *value.shape[1:]
            )  # [rollout_epoch, n_chunk_step, bsz, ...]
            new_value = new_value.transpose(
                0, 1
            )  # [n_chunk_step, rollout_epoch, bsz, ...]
            new_value = new_value.reshape(new_value.shape[0], -1, *new_value.shape[3:])
            ret_dict[key] = new_value
        elif isinstance(value, dict):
            ret_dict[key] = process_nested_dict_for_adv(value, rollout_epoch)
    return ret_dict


def process_nested_dict_for_train(nested_dict, shuffle_id):
    ret_dict = {}
    for key, value in nested_dict.items():
        if key in ['success']:
            continue
        if key in ["dones", "terminations", "truncations", "prev_values"]:
            value = value[:-1]
        if "env_info" in key:
            raise NotImplementedError
        if value is None:
            ret_dict[key] = None
        if isinstance(value, torch.Tensor):
            ret_dict[key] = value.reshape(-1, *value.shape[2:])[shuffle_id]
        elif isinstance(value, dict):
            ret_dict[key] = process_nested_dict_for_train(value, shuffle_id)
    return ret_dict


class FSDPActor(FSDPModelManager, Worker):
    def __init__(
        self, cfg: DictConfig, placement: ModelParallelComponentPlacement
    ) -> None:
        """
        FSDPActor worker used to train the model with data from rollout workers.

        Args:
            cfg (DictConfig): The global yaml configuration.
            placement (ModelParallelComponentPlacement): The accelerator placement for actor worker.
        """
        Worker.__init__(self)
        super().__init__(cfg.actor, self._world_size, self._rank)

        self.cfg = cfg

        self.response_len = (
            self.cfg.actor.model.encoder_seq_length - self.cfg.data.max_prompt_length
        )
        self.calculate_entropy = self.cfg.algorithm.calculate_entropy
        self.calculate_entropy_loss = (
            self.cfg.algorithm.entropy_bonus > 0 and self.calculate_entropy
        )
        self.kl_beta = self.cfg.algorithm.kl_beta
        self.kl_penalty_type = self.cfg.algorithm.kl_penalty_type

        self.total_batch_size_per_dp = (
            self.cfg.data.rollout_batch_size
            * self.cfg.algorithm.group_size
            // self._world_size
        )

        self._rollout_group_name = cfg.rollout.group_name
        self._component_placement = placement
        self.is_pipeline = self._component_placement.is_disaggregated
        self.ref_policy_state_dict = None
        if self.is_pipeline:
            self._inference_group_name = cfg.inference.group_name
            self._inference_world_size = self._component_placement.get_world_size(
                "inference"
            )
            self._inference_dst_map: dict[int, list[str]] = {}
        else:
            self._inference_group_name = None
            self._inference_world_size = 0
            self._inference_dst_map = None
        self.loss_agg_func = get_loss_agg_func(self.cfg.algorithm.loss_agg_func)
        self.enable_offload = (
            self.cfg.actor.get("enable_offload", False) and not self.is_pipeline
        )
        self.micro_batch_size = self.cfg.actor.micro_batch_size
        self.n_mini_batches = self.cfg.algorithm.n_minibatches
        self.task_type = self.cfg.runner.task_type
        self.entropy_op_type = self.cfg.algorithm.get("entropy_op_type", "liger_kernel")

    def init_worker(self) -> None:
        """
        Initialize the actor worker. build the model and use corresponding training backend
        (FSDP/FSDP2) to wrap it. If needed, offload model parameters and optimizer states to CPU.
        If kl_beta > 0, retrieve the reference policy model state dict to CPU.
        If mode is disaggregated, setup which inference ranks it needs to sync weights to by
        doing a handshake with inference workers.
        """
        self.setup_model_and_optimizer()
        if self.cfg.algorithm.kl_beta > 0 and self.cfg.actor.get(
            "combine_reference_model", True
        ):
            self.ref_policy_state_dict = retrieve_model_state_dict_in_cpu(self.model)

        if self.enable_offload and not self.is_pipeline:
            self.offload_param_and_grad()
            self.offload_optimizer()
        self._setup_rollout_weight_dst_ranks()

    def _setup_rollout_weight_dst_ranks(self) -> None:
        """Setup destination ranks for token and weight communication."""
        rank_map = RankMapper.get_actor_rank_to_rollout_rank_map(
            self._component_placement
        )
        self._weight_dst_rank_in_rollout = rank_map[self._rank]
        self.log_info(
            f"Actor rank {self._rank} will send weights to {self._weight_dst_rank_in_rollout}"
        )

    def del_reshard_state_dict(self) -> None:
        """Just for interface compatibility with MegatronActor."""
        if hasattr(self, "rollout_state_dict"):
            del self.rollout_state_dict
        clear_memory(sync=False)

    def sync_model_to_inference(self) -> None:
        """
        Sync the model's full state dict to the inference worker.
        The model state_dict is the reference of actor's model
        parameters(by setting cpu_offload=False).
        """
        if not self._inference_dst_map:
            self._strategy.setup_actor_sync_inference_ranks(self)

        if self.is_optimizer_offloaded:
            self.offload_optimizer()

        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device, False)

        inference_state_dict = self.get_model_state_dict(
            cpu_offload=False, full_state_dict=False
        )
        # NOTE: we have already know which inference rank needs which params
        # by calling _strategy.setup_actor_sync_inference_ranks() to do handshake
        # with each inference rank. just send them accordingly.
        for rank, needed_params in self._inference_dst_map.items():
            sended_params = {}
            for name in needed_params:
                if name in inference_state_dict:
                    # mentioned again, no ShardedTensor here.
                    sended_params[name] = (
                        inference_state_dict[name].to_local()
                        if isinstance(inference_state_dict[name], DTensor)
                        else inference_state_dict[name]
                    )
            self.send(
                object=sended_params,
                dst_group_name=self._inference_group_name,
                dst_rank=rank,
                async_op=True,
            )

        if self.enable_offload and not self.is_weight_offloaded:
            self.offload_param_and_grad()

        torch.distributed.barrier()

    def sync_model_to_rollout(self) -> None:
        """
        Sync the model's full state dict to the rollout worker.
        """
        if self.enable_offload and not self.is_optimizer_offloaded:
            self.offload_optimizer()

        if self.enable_offload and self.is_weight_offloaded:
            self.load_param_and_grad(self.device, True)

        self.rollout_state_dict = self.get_model_state_dict(
            cpu_offload=False, full_state_dict=True
        )

        has_visual = any("visual." in k for k in self.rollout_state_dict.keys())

        state_dict = {}

        if self._weight_dst_rank_in_rollout is not None:
            for k, v in self.rollout_state_dict.items():
                name = k
                if has_visual:
                    if name.startswith("model.language_model."):
                        name = "model." + name[21:]
                    # NOTE:
                    # if transformers version is 4.56.1 or older(not tested),
                    # the following line should be uncommented

                    # elif name.startswith("model."):
                    #     name = name[6:]
                state_dict[name] = reduce_tensor(v) if not self.is_pipeline else v
            if not self.is_pipeline:
                self.send(
                    state_dict,
                    self._rollout_group_name,
                    self._weight_dst_rank_in_rollout,
                )
            else:
                for weight_dst_rank in self._weight_dst_rank_in_rollout:
                    self.send(
                        state_dict,
                        self._rollout_group_name,
                        weight_dst_rank,
                    )

        state_dict.clear()
        if self.enable_offload and not self.is_weight_offloaded:
            self.offload_param_and_grad()

    def get_batch(
        self, channel: Channel
    ) -> tuple[dict[str, torch.Tensor], RolloutResult]:
        result: RolloutResult = channel.get()

        batch = result.to_actor_batch(
            self.cfg.data.max_prompt_length,
            self.cfg.actor.model.encoder_seq_length,
            self.tokenizer.eos_token_id,
        )
        return batch, result

    def _load_weight_and_optimizer(self) -> None:
        # Acquire the GPUs to ensure that no one is using them before loading models
        # Otherwise, it may lead to OOM
        with self.device_lock:
            if not self.enable_offload:
                return
            if self.is_weight_offloaded:
                self.load_param_and_grad(self.device)
            if self.is_optimizer_offloaded:
                self.load_optimizer(self.device)

    @torch.no_grad()
    def inference_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        self.model.eval()
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        position_ids = batch["position_ids"]

        multi_modal_inputs = {}
        if "multi_modal_inputs" in batch.keys():
            for key in batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat(
                    [inputs[key] for inputs in batch["multi_modal_inputs"]],
                    dim=0,
                ).cuda()

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            **multi_modal_inputs,
        )

        logits = outputs.logits
        logits = logits[:, -self.response_len - 1 : -1, :]
        logits = logits / self.cfg.algorithm.sampling_params.temperature

        responses = input_ids[:, -self.response_len :]
        logprobs = compute_logprobs_from_logits(
            logits=logits, target=responses, op_type=self.entropy_op_type
        )
        return logprobs

    def run_inference(
        self,
        input_channel: Channel,
        output_channel: Channel,
        compute_ref_logprobs: bool,
    ) -> None:
        """
        Compute prev/ref logprobs using the actor Model's forward.

        Args:
            input_channel: The input channel to read from.
            output_channel: The output channel to send results to.
            compute_ref_logprobs: Whether to compute reference logprobs.
        """
        recv_batch_size = 0
        while recv_batch_size < self.total_batch_size_per_dp:
            batch, rollout_result = self.get_batch(input_channel)
            recv_batch_size += rollout_result.num_sequence
            self._load_weight_and_optimizer()

            num_splits = (
                rollout_result.num_sequence
                // self.cfg.algorithm.logprob_forward_micro_batch_size
            )
            micro_batches_iter = get_iterator_k_split(
                batch,
                num_splits=num_splits,
            )
            micro_batches = list(micro_batches_iter)

            prev_logprobs = []
            with self.worker_timer():
                for micro_batch in micro_batches:
                    prev_logprobs.append(self.inference_step(micro_batch).cpu())

                if rollout_result.rollout_logprobs is not None:
                    # Rollout has returned logprobs, store the recomputed logprobs in recompute_prev_logprobs
                    rollout_result.recompute_prev_logprobs = torch.cat(prev_logprobs)
                else:
                    # Otherwise, directly store the logprobs in prev_logprobs (the final logprobs used for training)
                    rollout_result.prev_logprobs = torch.cat(prev_logprobs)

            if compute_ref_logprobs:
                assert self.ref_policy_state_dict is not None, (
                    "Reference policy state dict is None but compute_ref_logprobs is True"
                )
                ref_logprobs = []
                with cpu_weight_swap(self.model, self.ref_policy_state_dict):
                    for micro_batch in micro_batches:
                        ref_logprobs.append(self.inference_step(micro_batch).cpu())
                    rollout_result.ref_logprobs = torch.cat(ref_logprobs)

            output_channel.put(rollout_result)

        assert recv_batch_size == self.total_batch_size_per_dp, (
            f"Expected {self.total_batch_size_per_dp} sequences from channel, but got {recv_batch_size}"
        )

    def training_step(
        self, batch: dict[str, torch.Tensor] | BatchResizingIterator
    ) -> tuple[dict[str, torch.Tensor], float, list[float]]:
        if isinstance(batch, dict):
            global_batch_size = batch["input_ids"].shape[0]
            assert global_batch_size % self.micro_batch_size == 0, (
                f"global batch size {global_batch_size} can not divide micro_batch_size {self.micro_batch_size}"
            )
            micro_batch_cnt = global_batch_size // self.micro_batch_size
            self.gradient_accumulation = micro_batch_cnt
            micro_batches = get_iterator_k_split(batch, micro_batch_cnt)
            micro_batches_iter = iter(micro_batches)
        else:
            global_batch_size = self.total_batch_size_per_dp // self.n_mini_batches
            micro_batch_cnt = global_batch_size // self.micro_batch_size
            self.gradient_accumulation = micro_batch_cnt

            def iterator_wrapper():
                for _ in range(micro_batch_cnt):
                    yield next(batch)

            micro_batches_iter = iterator_wrapper()
        self.optimizer.zero_grad()
        mbs_metrics_list = {}
        for idx, m_batch in enumerate(micro_batches_iter):
            backward_ctx = self.before_micro_batch(
                self.model,
                is_last_micro_batch=(idx + 1) == self.gradient_accumulation,
            )
            for k, v in m_batch.items():
                m_batch[k] = v.cuda() if isinstance(v, torch.Tensor) else v

            multi_modal_inputs = {}
            if "multi_modal_inputs" in m_batch.keys():
                for key in m_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = torch.cat(
                        [inputs[key] for inputs in m_batch["multi_modal_inputs"]],
                        dim=0,
                    ).cuda()

            input_ids = m_batch["input_ids"]
            attention_mask = m_batch["attention_mask"]
            position_ids = m_batch["position_ids"]
            prev_logprobs = m_batch["prev_logprobs"]
            advantages = m_batch["advantages"]
            ref_logprobs = None
            if "ref_logprobs" in m_batch:
                ref_logprobs = m_batch["ref_logprobs"]

            loss_mask = m_batch["response_mask"][:, -self.response_len :]

            clip_ratio = self.cfg.algorithm.ratio_clip_eps
            clip_ratio_low = self.cfg.algorithm.get("clip_ratio_low", None)
            clip_ratio_high = self.cfg.algorithm.get("clip_ratio_high", None)
            clip_ratio_low = (
                clip_ratio_low if clip_ratio_low is not None else clip_ratio
            )
            clip_ratio_high = (
                clip_ratio_high if clip_ratio_high is not None else clip_ratio
            )
            clip_ratio_c = self.cfg.algorithm.get("clip_ratio_c", 3.0)

            with self.amp_context:
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                )

                logits: torch.Tensor = output.logits

                logits.div_(self.cfg.algorithm.sampling_params.temperature)

                responses = input_ids[:, -self.response_len :]
                logits = logits[
                    :, -self.response_len - 1 : -1, :
                ]  # (bsz, response_length, vocab_size)
                logprobs = compute_logprobs_from_logits(
                    logits, responses, self.entropy_op_type
                )

                if self.cfg.algorithm.get("importance_sampling_fix", False):
                    rollout_prev_logprobs = prev_logprobs
                    recompute_prev_logprobs = m_batch["recompute_prev_logprobs"]
                    advantages = advantages * torch.clamp(
                        (recompute_prev_logprobs - rollout_prev_logprobs).exp(),
                        min=self.cfg.algorithm.importance_sampling_clip,
                    )

                loss, mbs_metrics_data = policy_loss(
                    loss_type=self.cfg.algorithm.loss_type,
                    loss_agg_func=self.loss_agg_func,
                    logprobs=logprobs,
                    old_logprobs=prev_logprobs,
                    advantages=advantages,
                    clip_ratio_low=clip_ratio_low,
                    clip_ratio_high=clip_ratio_high,
                    clip_ratio_c=clip_ratio_c,
                    loss_mask=loss_mask,
                    task_type=self.task_type,
                )

                entropy_loss = torch.tensor(0.0, device=torch.cuda.current_device())
                if self.calculate_entropy:
                    entropy = compute_entropy_from_logits(
                        logits,
                    )

                    entropy_loss = self.loss_agg_func(entropy, mask=loss_mask)
                    if self.calculate_entropy_loss:
                        loss = loss - self.cfg.algorithm.entropy_bonus * entropy_loss

                kl_loss = torch.tensor(0.0, device=torch.cuda.current_device())
                if self.kl_beta > 0 and ref_logprobs is not None:
                    kld = kl_penalty(ref_logprobs, logprobs, self.kl_penalty_type)
                    kl_loss = self.loss_agg_func(kld, loss_mask)
                    loss = loss + kl_loss * self.kl_beta

                # add to log
                # scale loss for gradient accumulation and backprop
                loss = loss / self.gradient_accumulation
                with backward_ctx:
                    self.grad_scaler.scale(loss).backward()

            mbs_metrics_data.update(
                {
                    "actor/final_loss": loss.detach(),
                    "actor/entropy_loss": entropy_loss.detach(),
                    "actor/kl_loss": kl_loss.detach(),
                }
            )

            append_to_dict(mbs_metrics_list, mbs_metrics_data)

        grad_norm, lr_list = self.optimizer_step()
        return mbs_metrics_list, grad_norm, lr_list

    def run_training_pipeline(self, input_channel: Channel) -> tuple[dict, list]:
        self.model.train()
        train_batch_iterator = BatchResizingIterator(
            cfg=self.cfg,
            get_batch_fn=partial(self.get_batch, input_channel),
            micro_batch_size=self.micro_batch_size,
            total_batch_size=self.total_batch_size_per_dp,
            num_global_batches=self.n_mini_batches,
            forward_only=False,
        )
        train_batch_iterator.register_get_batch_handler(
            self.compute_advantages_and_returns
        )

        if self.cfg.algorithm.normalize_advantages:

            def normalize_advantages(batch: dict[str, torch.Tensor]):
                mask = batch["response_mask"][:, -self.response_len :]
                batch["advantages"] = masked_normalization(batch["advantages"], mask)
                return batch

            train_batch_iterator.register_global_batch_handler(normalize_advantages)

        self._load_weight_and_optimizer()
        training_metrics_list = []
        with self.worker_timer():
            for _ in range(self.n_mini_batches):
                metrics, grad_norm, lr_list = self.training_step(
                    batch=train_batch_iterator
                )

                # aggregate metrics across micro-batches
                mean_metric_dict = {
                    key: torch.mean(torch.stack(value))
                    for key, value in metrics.items()
                }
                mean_metric_dict = all_reduce_dict(
                    mean_metric_dict, op=torch.distributed.ReduceOp.AVG
                )

                mean_metric_dict["actor/grad_norm"] = float(grad_norm)
                mean_metric_dict["actor/lr"] = lr_list[0]
                training_metrics_list.append(mean_metric_dict)

        # put lr scheduler step here
        self.lr_scheduler.step()

        # Rollout metrics
        batch = train_batch_iterator.get_all_batches()
        rollout_metrics, _, _ = compute_math_rollout_metrics(
            batch, self.cfg.data.max_prompt_length, self.response_len
        )

        return rollout_metrics, training_metrics_list

    def run_training(self, input_channel: Channel) -> tuple[dict, list]:
        # Get all batches for this DP
        if self.is_pipeline:
            with self.worker_timer():
                return self.run_training_pipeline(input_channel)

        batches = []
        recv_batch_size = 0
        while recv_batch_size < self.total_batch_size_per_dp:
            batch, rollout_result = self.get_batch(input_channel)
            batches.append(batch)
            recv_batch_size += rollout_result.num_sequence
        assert recv_batch_size == self.total_batch_size_per_dp, (
            f"Expected {self.total_batch_size_per_dp} sequences from channel, but got {recv_batch_size}"
        )
        global_batch = RolloutResult.merge_batches(batches)

        # Compute advantages and returns
        global_batch = self.compute_advantages_and_returns(global_batch)

        if self.cfg.algorithm.normalize_advantages:
            mask = global_batch["response_mask"][:, -self.response_len :]
            global_batch["advantages"] = masked_normalization(
                global_batch["advantages"], mask
            )

        # Must be called after batch is retrieved, which is when rollout has stopped
        # Otherwise, loading model might cause OOM
        self._load_weight_and_optimizer()

        mini_batches = get_iterator_k_split(
            global_batch,
            num_splits=self.cfg.algorithm.n_minibatches,
            shuffle=self.cfg.algorithm.get("shuffle_rollout", True),
            shuffle_seed=self.cfg.actor.seed,
        )

        self.model.train()
        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        )

        training_metrics_list = []
        # Global batch iterations
        with self.worker_timer():
            for mini_batch in mini_batches:
                metrics, grad_norm, lr_list = self.training_step(batch=mini_batch)

                # aggregate metrics across micro-batches
                mean_metric_dict = {
                    key: torch.mean(torch.stack(value))
                    for key, value in metrics.items()
                }
                mean_metric_dict = all_reduce_dict(
                    mean_metric_dict, op=torch.distributed.ReduceOp.AVG
                )

                mean_metric_dict["actor/grad_norm"] = float(grad_norm)
                mean_metric_dict["actor/lr"] = lr_list[0]
                training_metrics_list.append(mean_metric_dict)

        # put lr scheduler step here
        self.lr_scheduler.step()

        # Rollout metrics
        rollout_metrics, _, _ = compute_math_rollout_metrics(
            global_batch, self.cfg.data.max_prompt_length, self.response_len
        )

        return rollout_metrics, training_metrics_list

    # Advantages and returns
    def compute_advantages_and_returns(self, batch: dict[str, torch.Tensor]):
        """Compute the advantages and returns.

        Args:
            batch (Dict[str, torch.Tensor]): The rollout batch.
        """
        with self.worker_timer():
            if batch.get("advantages", None) is None:
                mask = batch["response_mask"][:, -self.response_len :]
                advantages, _ = calculate_adv_and_returns(
                    task_type=self.task_type,
                    adv_type=self.cfg.algorithm.adv_type,
                    rewards=batch["rewards"].cuda(),
                    loss_mask=mask.cuda(),
                    group_size=self.cfg.algorithm.group_size,
                    kl_beta=self.cfg.algorithm.get("reinpp_kl_beta", 0.0),
                    kl_penalty_type=self.kl_penalty_type,
                    logprob=batch["prev_logprobs"].cuda()
                    if "prev_logprobs" in batch
                    else None,
                    ref_logprob=batch["ref_logprobs"].cuda()
                    if "ref_logprobs" in batch
                    else None,
                    use_reinpp_baseline=self.cfg.algorithm.get(
                        "use_reinpp_baseline", False
                    ),
                )
                batch["advantages"] = advantages

        return batch


class EmbodiedFSDPActor(FSDPModelManager, Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        super().__init__(cfg.actor, self._world_size, self._rank)
        self.cfg = cfg
        self._env_group_name = cfg.env.group_name
        self._rollout_group_name = cfg.rollout.group_name
        self._component_placement = HybridComponentPlacement(cfg, Cluster())

        # stage_num: default to 2, use for pipeline rollout process
        self.stage_num = cfg.rollout.pipeline_stage_num

        self.enable_offload = self.cfg.actor.get("enable_offload", False)
        self.entropy_op_type = self.cfg.algorithm.get("entropy_op_type", "torch")

        # Sync weight comm options
        max_ctas = cfg.rollout.get("sync_weight_nccl_max_ctas", None)
        min_ctas = cfg.rollout.get("sync_weight_nccl_min_ctas", None)
        self._sync_weight_comm_options = CollectiveGroupOptions(
            accel_max_ctas=max_ctas, accel_min_ctas=min_ctas
        )
        
        self.intrinsic_cfg = self.cfg.reward.get("intrinsic_reward", None)
        self.use_intrinsic_reward = bool(self.intrinsic_cfg)
        self.intrinsic_coef = float(self.intrinsic_cfg.get("coef", 1.0))if self.use_intrinsic_reward else 0.0
        self._phi_heads_loaded = False
        if self.use_intrinsic_reward:
            self.goal_ema_decay = float(self.intrinsic_cfg.get("goal_ema_decay", 0.99))
        self.goal_ema_table: Optional[torch.Tensor] = None

    def _setup_rollout_weight_dst_ranks(self) -> None:
        """
        Setup destination ranks for weight communication.
        It can support any topology between actor and rollout workers.
        Assuming there are M actor ranks and N rollout ranks, each actor rank
        will send weights to most ceil(N/M) rollout ranks according to the modulo rule.
        """
        rollout_world_size = self._component_placement.get_world_size("rollout")
        actor_world_size = self._world_size
        rank = self._rank
        self._weight_dst_rank_in_rollout = []
        rollout_ranks_per_actor = (
            rollout_world_size + actor_world_size - 1
        ) // actor_world_size
        for i in range(rollout_ranks_per_actor):
            if i * actor_world_size + rank < rollout_world_size:
                self._weight_dst_rank_in_rollout.append(i * actor_world_size + rank)

    def init_worker(self) -> None:
        """
        Initialize the actor worker. build the model and use corresponding training backend,
        if needed, offload model parameters and optimizer states to CPU.
        """
        self.setup_model_and_optimizer()

        if self.enable_offload:
            self.offload_param_and_grad()
            self.offload_optimizer()
        self._setup_rollout_weight_dst_ranks()

    def _fsdp_full_params_ctx(self, module=None, writeback: bool = False):
        if module is None:
            module = self.model
        from rlinf.hybrid_engines.fsdp import FSDP
        try:
            return FSDP.summon_full_params(module, writeback=writeback, recurse=False)
        except TypeError:
            return FSDP.summon_full_params(module, writeback=writeback)

    def _soft_update_target_phi(self) -> None:
        tau = self.intrinsic_cfg.phi_target_tau
        with torch.no_grad():
            with self._fsdp_full_params_ctx(self.model.target_phi_head, writeback=True):
                with self._fsdp_full_params_ctx(self.model.phi_head, writeback=False):
                    for target_param, src_param in zip(self.model.target_phi_head.parameters(),self.model.phi_head.parameters()):
                        target_param.data.mul_(1.0 - tau)
                        target_param.data.add_(tau * src_param.data)



    def _load_phi_heads_into_module(self, module):
        checkpoint_path = self.intrinsic_cfg.phi_checkpoint
        state = torch.load(checkpoint_path, map_location="cpu")
        for attr in ("phi_head", "target_phi_head"):
            head = getattr(module, attr)
            head.load_state_dict(state[attr], strict=False)
            head.to(self.device)
        self.goal_ema_table = state["goal_phi_table"].to(self.device)
        self._phi_heads_loaded = True

    def _ema_update(self, prev: Optional[torch.Tensor], new: torch.Tensor) -> torch.Tensor:
        if prev is None:
            return new
        return self.goal_ema_decay * prev + (1.0 - self.goal_ema_decay) * new

    def _update_goal_ema(self, task_ids: Optional[torch.Tensor], phi_goal: torch.Tensor) -> None:
        for task_id in torch.unique(task_ids):
            mask = task_ids == task_id
            mean_feat = phi_goal[mask].mean(dim=0)
            key = int(task_id.item())
            prev = self.goal_ema_table[key]
            self.goal_ema_table[key] = self._ema_update(prev, mean_feat)
    
    def _compute_intrinsic_from_features(self, rollout_batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute intrinsic potential directly from cached phi features without encoder forward.
        """
        phi_dtype = self.model.phi_head.phi_list[0].fc1.weight.dtype
        state_feat = rollout_batch["phi_state_features"].to(self.device, dtype=phi_dtype, non_blocking=True)
        goal_feat = rollout_batch["phi_goal_features"].to(self.device, dtype=phi_dtype, non_blocking=True)

        intrinsic_goal_mask = rollout_batch["phi_suc_mask"].to(self.device, dtype=torch.bool)
        task_ids = rollout_batch["task_ids"]
        task_ids = task_ids[0].view(-1).to(self.device, dtype=torch.long, non_blocking=True)
        fail_mask = (~intrinsic_goal_mask)
        phi_state = self.model.phi_head(state_feat, return_phi=True)
        phi_goal = self.model.phi_head(goal_feat, return_phi=True)
        if fail_mask.any():
            phi_default_goal = self.goal_ema_table[task_ids[fail_mask]]
            phi_goal[:, fail_mask, :, :] = phi_default_goal.to(dtype=phi_goal.dtype)
        v_now = compute_v_sg(phi_state, phi_goal).mean(dim=-1)
        #TODO: can add more transform to get reasonable intrinsic value range:
        if self.intrinsic_cfg.get("intrinsic_reward_transform_type", False):
            int_r = self.transform_v_function(v_now, fail_mask)
        else:
            int_r =  v_now

        if intrinsic_goal_mask.any():
            self._update_goal_ema(task_ids[intrinsic_goal_mask], phi_goal[0, intrinsic_goal_mask, :, :])

        # rollout_batch.pop("phi_state_features")
        # rollout_batch.pop("phi_suc_features")
        rollout_batch.pop("phi_suc_mask")
        return int_r.detach().cpu()

    def compute_intrinsic_rewards(self, rollout_batch: dict[str, torch.Tensor]):
        intrinsic_reward = self._compute_intrinsic_from_features(rollout_batch)
        rewards = rollout_batch["rewards"]
        chunk_size = rewards.shape[-1]
        shaped = intrinsic_reward.unsqueeze(-1).expand(-1, -1, chunk_size).clone()
        shaped[:, :, 1:] = 0.
        mask = rollout_batch.get("loss_mask", None)
        if mask is not None:
            shaped = shaped * mask
        shaped = shaped.to(rewards.dtype)
        rollout_batch["intrinsic_rewards"] = shaped
        rollout_batch["original_rewards"] = rewards
        rollout_batch["rewards"] = rewards + self.intrinsic_coef * shaped

    def _prepare_intrinsic_features_from_rollout(
        self, rollout_batch: dict[str, torch.Tensor]
    ) -> bool:
        """
        Use phi features cached during rollout (if available) to avoid re-forwarding the encoder.
        """
        phi_features = rollout_batch['forward_inputs']['phi_features']
        rollout_batch["task_ids"] = rollout_batch['forward_inputs']["task_ids"]
        if phi_features is None:
            return False
        dones_tensor = rollout_batch["dones"]
        n_step, batch_size = phi_features.shape[0], phi_features.shape[1]
        task_ids = rollout_batch['forward_inputs']["task_ids"]
        task_ids = task_ids[0].view(-1)
        task_ids = task_ids.to(phi_features.device)

        # next-step features with episode boundary handling
        dones_step = dones_tensor.any(dim=-1)
        actual_dones_step = dones_step[:-1]
        boundary = actual_dones_step.unsqueeze(-1)  # [n_step-1, bsz, 1]
        next_phi = torch.roll(phi_features, shifts=-1, dims=0)
        next_phi = torch.where(boundary, phi_features, next_phi)

        # goal features: first done per trajectory (fallback to last step)
        step_idx = torch.arange(n_step, device=phi_features.device).view(-1, 1)
        masked_idx = torch.where(actual_dones_step, step_idx, torch.full_like(step_idx, n_step))
        first_done_idx = masked_idx.min(dim=0).values  #[bsz,]
        fallback_idx = torch.full_like(first_done_idx, n_step - 1)
        last_done_idx = torch.where(first_done_idx < n_step, first_done_idx, fallback_idx)
        batch_indices = torch.arange(batch_size, device=phi_features.device)
        goal_feat = phi_features.transpose(0, 1)[batch_indices, last_done_idx]

        rollout_batch["phi_state_features"] = phi_features.contiguous()
        rollout_batch["phi_next_features"] = next_phi.contiguous()
        rollout_batch["phi_goal_features"] = goal_feat.unsqueeze(0).expand_as(phi_features).contiguous()
        step_to_goal = torch.clamp(last_done_idx.unsqueeze(0) - step_idx, min=0).unsqueeze(-1).float() 
        rollout_batch["step_to_goal"] = step_to_goal

        # success_tensor = rollout_batch['success']
        # success_tensor = rollout_batch['success'].to(phi_features.device)
        # success_step = (success_tensor.any(dim=-1))[:-1]
        # masked_success = torch.where(success_step, step_idx, torch.full_like(step_idx, n_step))
        # first_success_idx = masked_success.min(dim=0).values
        # last_success_idx = torch.where(first_success_idx < n_step, first_success_idx, fallback_idx)
        # success_feat = phi_features.transpose(0, 1)[batch_indices, last_success_idx]
        # success_mask = (first_success_idx < n_step)
        # rollout_batch["episode_success"] = episode_success.unsqueeze(0).unsqueeze(-1).repeat(n_step, 1, dones_tensor.shape[-1])
        # rollout_batch["phi_suc_features"] = success_feat
        # rollout_batch["phi_suc_mask"] = success_mask

        # TODO: currently the done info is equal to successs info !
        

        episode_success = first_done_idx < n_step
        rollout_batch["episode_success"] = episode_success.unsqueeze(0).unsqueeze(-1).repeat(n_step, 1, dones_tensor.shape[-1])
        # rollout_batch["phi_suc_features"] = rollout_batch["phi_goal_features"]
        rollout_batch["phi_suc_mask"] = episode_success
        return True

    def model_provider_func(self) -> nn.Module:
        model = get_model(self.cfg.actor.model)
        if model is None:
            model = super().model_provider_func()
        if self.use_intrinsic_reward:
            self._load_phi_heads_into_module(model)
        if self.cfg.runner.get("ckpt_path", None):
            model_dict = torch.load(self.cfg.runner.ckpt_path)
            model.load_state_dict(model_dict)

        return model

    def save_checkpoint(self, save_path: str, global_steps: int) -> None:
        super().save_checkpoint(save_path, global_steps)
        if self.use_intrinsic_reward and self.goal_ema_table is not None and self._rank == 0:
            torch.save({
                    "goal_ema_table": self.goal_ema_table.cpu(),
                    "goal_ema_decay": self.goal_ema_decay,
                },
                os.path.join(save_path, "goal_ema_table.pt"),
            )

    def load_checkpoint(self, load_path: str) -> None:
        super().load_checkpoint(load_path)
        if self.use_intrinsic_reward:
            goal_ema_path = os.path.join(load_path, "goal_ema_table.pt")
            state = torch.load(goal_ema_path, map_location="cpu")
            table = state.get("goal_ema_table", None)
            self.goal_ema_table = table.to(self.device)
            self.goal_ema_decay = float(state.get("goal_ema_decay", self.goal_ema_decay))

    def transform_v_function(self, v_sg: torch.tensor, fail_mask:torch.tensor) -> torch.tensor:
        # v_sg: traj_len * batch_sz
        breakpoint()
        trans_type = self.intrinsic_cfg.get("intrinsic_reward_transform_type", 'default')
        if trans_type=='default':
            return v_sg
        elif trans_type=='scalar_tanh_-1_to_1':
            # return 0.1 * torch.tanh(v_sg/20.)
            return 0.1 * torch.tanh(v_sg/100.)
        elif trans_type=='exp_scalar_tanh_-1_to_1':
            r = torch.tanh(v_sg/100.).max(dim=0).values
            r = torch.exp(r)
            r[~fail_mask] = 0
            r = r.unsqueeze(0).expand_as(v_sg)
            return r
        elif trans_type=='scalar_tanh_0_to_1_total':
            r = torch.tanh(v_sg.to(torch.float32).max(dim=0)[0])
            r[~fail_mask] = 0
            r = r.unsqueeze(0).expand_as(v_sg)
            r = r/64.
            return r 
        
    def sync_model_to_rollout(self) -> None:
        """
        Sync the model's full state dict to the rollout worker.
        """
        if self.enable_offload and not self.is_optimizer_offloaded:
            self.offload_optimizer()

        if self.enable_offload and self.is_weight_offloaded:
            self.load_param_and_grad(self.device)

        state_dict = self.get_model_state_dict(cpu_offload=False, full_state_dict=True)
        for rank in self._weight_dst_rank_in_rollout:
            self.send(
                state_dict,
                self._rollout_group_name,
                rank,
                async_op=True,
                options=self._sync_weight_comm_options,
            )
        if self.enable_offload and not self.is_weight_offloaded:
            self.offload_param_and_grad()

    async def recv_rollout_trajectories(self, input_channel: Channel) -> None:
        """
        Receive rollout trajectories from rollout workers.

        Args:
            input_channel: The input channel to read from.
        """
        send_num = self._component_placement.get_world_size("rollout") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)

        recv_list = []
        for _ in range(split_num):
            trajectory: Trajectory = await input_channel.get(async_op=True).async_wait()
            recv_list.append(trajectory)

        self.rollout_batch = convert_trajectories_to_batch(recv_list)

        self.rollout_batch = self._process_received_rollout_batch(self.rollout_batch)

    def _process_received_rollout_batch(
        self, rollout_batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        original shape: [rollout_epoch x n_chunk_steps, bsz, num_action_chunks, ...]
        target shape: [n_chunk_steps, rollout_epoch x bsz, num_action_chunks, ...]
        """
        rollout_epoch = self.cfg.algorithm.rollout_epoch
        rollout_batch = process_nested_dict_for_adv(rollout_batch, rollout_epoch)
        
        if self.use_intrinsic_reward:
            # Prefer cached phi features from rollout to avoid extra forward;
            # fallback to reconstructing inputs if absent.
            self._prepare_intrinsic_features_from_rollout(rollout_batch)
            
        if (
            not self.cfg.env.train.auto_reset
            and not self.cfg.env.train.ignore_terminations
        ):
            dones = rollout_batch[
                "dones"
            ]  # [n_chunk_step, rollout_epoch x bsz, num_action_chunks]
            loss_mask, loss_mask_sum = compute_loss_mask(dones)

            if self.cfg.algorithm.reward_type == "chunk_level":
                loss_mask = loss_mask.any(dim=-1, keepdim=True)
                loss_mask_sum = loss_mask_sum[..., -1:]

            rollout_batch["loss_mask"] = loss_mask
            rollout_batch["loss_mask_sum"] = loss_mask_sum

        # filter data by rewards
        if self.cfg.algorithm.get("filter_rewards", False):
            rewards = rollout_batch[
                "rewards"
            ]  # [n_chunk_step, batch, num_action_chunks]
            if rollout_batch.get("loss_mask", None) is not None:
                rewards = rewards * rollout_batch["loss_mask"]
            n_chunk_step, batch_size, num_action_chunks = rewards.shape

            group_size = self.cfg.algorithm.group_size
            assert batch_size % group_size == 0, (
                f"batch {batch_size} not divisible by group_size {group_size}"
            )
            n_prompts = batch_size // group_size

            # calculate rewards by prompt
            rewards = rewards.transpose(
                0, 1
            )  # [batch, n_chunk_step, num_action_chunks]
            rewards = rewards.reshape(rewards.shape[0], -1)  # [batch, n_step]
            reward_matrix = rewards.reshape(
                n_prompts, group_size, rewards.shape[-1]
            )  # [n_prompts, group_size, n_step]
            reward_matrix = reward_matrix.sum(dim=-1)  # [n_prompts, group_size]
            mean_reward_in_group = reward_matrix.mean(dim=1)  # [n_prompts]

            # mask
            reward_filter_mask = (
                mean_reward_in_group >= self.cfg.algorithm.rewards_lower_bound
            ) & (
                mean_reward_in_group <= self.cfg.algorithm.rewards_upper_bound
            )  # [n_prompts]

            # extend mask dimension
            reward_filter_mask = reward_filter_mask.repeat_interleave(
                group_size
            )  # [batch]
            reward_filter_mask = (
                reward_filter_mask.unsqueeze(0).expand(n_chunk_step, -1).unsqueeze(-1)
            )  # [n_chunk_step, batch, 1]

            # update loss_mask
            if rollout_batch.get("loss_mask", None) is not None:
                rollout_batch["loss_mask"] = (
                    reward_filter_mask & rollout_batch["loss_mask"]
                )
            else:
                rollout_batch["loss_mask"] = reward_filter_mask
        
        if self.use_intrinsic_reward:
            # When offload is enabled, parameters (including phi heads) may still be on CPU
            # because we offload right after init_worker. Load them back before using phi_head
            # to avoid device mismatch with rollout features on GPU.
            if self.enable_offload and self.is_weight_offloaded:
                self.load_param_and_grad(self.device)
            self.compute_intrinsic_rewards(rollout_batch)
        return rollout_batch

    def compute_advantages_and_returns(self) -> dict[str, torch.Tensor]:
        """
        Compute the advantages and returns.
        """
        kwargs = {
            "task_type": self.cfg.runner.task_type,
            "adv_type": self.cfg.algorithm.adv_type,
            "rewards": self.rollout_batch["rewards"],
            "dones": self.rollout_batch["dones"],
            "values": self.rollout_batch.get("prev_values", None),
            "gamma": self.cfg.algorithm.get("gamma", 1),
            "gae_lambda": self.cfg.algorithm.get("gae_lambda", 1),
            "group_size": self.cfg.algorithm.get("group_size", 8),
            "reward_type": self.cfg.algorithm.reward_type,
            "loss_mask": self.rollout_batch.get("loss_mask", None),
            "loss_mask_sum": self.rollout_batch.get("loss_mask_sum", None),
        }

        advantages_and_returns = calculate_adv_and_returns(**kwargs)

        self.rollout_batch.update(advantages_and_returns)
        if kwargs["loss_mask"] is not None:
            self.rollout_batch.update({"loss_mask": kwargs["loss_mask"]})
        if kwargs["loss_mask_sum"] is not None:
            self.rollout_batch.update({"loss_mask_sum": kwargs["loss_mask_sum"]})

        rollout_metrics = compute_rollout_metrics(self.rollout_batch)
        return rollout_metrics

    def run_training(self) -> None:
        """
        Run the training process using the received rollout batch.
        """
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)

        self.model.train()
        rollout_size = (
            self.rollout_batch["prev_logprobs"].shape[0]
            * self.rollout_batch["prev_logprobs"].shape[1]
        )
        g = torch.Generator()
        g.manual_seed(self.cfg.actor.seed + self._rank)
        shuffle_id = torch.randperm(rollout_size, generator=g)

        with torch.no_grad():
            self.rollout_batch = process_nested_dict_for_train(
                self.rollout_batch, shuffle_id
            )

        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        ), "global_batch_size is not divisible by micro_batch_size * world_size"

        self.gradient_accumulation = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        rollout_size = self.rollout_batch["prev_logprobs"].size(0)
        batch_size_per_rank = self.cfg.actor.global_batch_size // self._world_size
        assert rollout_size % batch_size_per_rank == 0, (
            f"{rollout_size} is not divisible by {batch_size_per_rank}"
        )
        metrics = {}
        update_epoch = self.cfg.algorithm.get("update_epoch", 1)
        for _ in range(update_epoch):
            rollout_dataloader_iter = split_dict_to_chunk(
                self.rollout_batch,
                rollout_size // batch_size_per_rank,
            )
            for train_global_batch in rollout_dataloader_iter:
                # split batch into micro_batches
                train_global_batch_size = train_global_batch["prev_logprobs"].shape[0]
                assert (
                    train_global_batch_size
                    == self.cfg.actor.global_batch_size
                    // torch.distributed.get_world_size()
                )
                assert train_global_batch_size % self.cfg.actor.micro_batch_size == 0, (
                    f"{train_global_batch_size=}, {self.cfg.actor.micro_batch_size}"
                )

                train_micro_batch = split_dict_to_chunk(
                    train_global_batch,
                    train_global_batch_size // self.cfg.actor.micro_batch_size,
                )

                self.optimizer.zero_grad()
                for idx, batch in enumerate(train_micro_batch):
                    batch = put_tensor_device(
                        batch, f"cuda:{int(os.environ['LOCAL_RANK'])}"
                    )
                    backward_ctx = self.before_micro_batch(
                        self.model,
                        is_last_micro_batch=(idx + 1) == self.gradient_accumulation,
                    )
                    advantages = batch["advantages"]
                    prev_logprobs = batch["prev_logprobs"]
                    returns = batch.get("returns", None)
                    prev_values = batch.get("prev_values", None)
                    loss_mask = batch.get("loss_mask", None)
                    loss_mask_sum = batch.get("loss_mask_sum", None)

                    forward_inputs = batch.get("forward_inputs", None)

                    kwargs = {}
                    if SupportedModel(self.cfg.actor.model.model_type) in [
                        SupportedModel.OPENVLA,
                        SupportedModel.OPENVLA_OFT,
                    ]:
                        kwargs["temperature"] = (
                            self.cfg.algorithm.sampling_params.temperature_train
                        )
                        kwargs["top_k"] = self.cfg.algorithm.sampling_params.top_k
                    elif (
                        SupportedModel(self.cfg.actor.model.model_type)
                        == SupportedModel.GR00T
                    ):
                        kwargs["prev_logprobs"] = prev_logprobs

                    compute_values = (
                        True if self.cfg.algorithm.adv_type == "gae" else False
                    )
                    with self.amp_context:
                        output_dict = self.model(
                            forward_inputs=forward_inputs,
                            compute_logprobs=True,
                            compute_entropy=self.cfg.algorithm.entropy_bonus > 0,
                            compute_values=compute_values,
                            use_cache=False,
                            **kwargs,
                        )

                    if (
                        SupportedModel(self.cfg.actor.model.model_type)
                        == SupportedModel.GR00T
                    ):
                        prev_logprobs = output_dict["prev_logprobs"]

                    kwargs = {
                        "loss_type": self.cfg.algorithm.loss_type,
                        "logprob_type": self.cfg.algorithm.logprob_type,
                        "reward_type": self.cfg.algorithm.reward_type,
                        "single_action_dim": self.cfg.actor.model.get("action_dim", 7),
                        "logprobs": output_dict["logprobs"],
                        "values": output_dict.get("values", None),
                        "old_logprobs": prev_logprobs,
                        "advantages": advantages,
                        "returns": returns,
                        "prev_values": prev_values,
                        "clip_ratio_high": self.cfg.algorithm.clip_ratio_high,
                        "clip_ratio_low": self.cfg.algorithm.clip_ratio_low,
                        "value_clip": self.cfg.algorithm.get("value_clip", None),
                        "huber_delta": self.cfg.algorithm.get("huber_delta", None),
                        "loss_mask": loss_mask,
                        "loss_mask_sum": loss_mask_sum,
                        "max_episode_steps": self.cfg.env.train.max_episode_steps,
                        "task_type": self.cfg.runner.task_type,
                        "critic_warmup": self.optimizer_steps
                        < self.critic_warmup_steps,
                    }
                    loss, metrics_data = policy_loss(**kwargs)

                    entropy_loss = torch.tensor(0.0, device=torch.cuda.current_device())
                    if (
                        self.cfg.algorithm.entropy_bonus > 0
                        and not kwargs["critic_warmup"]
                    ):
                        entropy = output_dict["entropy"]
                        entropy = reshape_entropy(
                            entropy,
                            entropy_type=self.cfg.algorithm.entropy_type,
                            action_dim=self.cfg.actor.model.get("action_dim", 7),
                            batch_size=output_dict["logprobs"].shape[0],
                        )
                        entropy_loss = masked_mean(entropy, mask=loss_mask)
                        loss -= self.cfg.algorithm.entropy_bonus * entropy_loss
                    metrics_data["actor/entropy_loss"] = entropy_loss.detach().item()
                    if self.use_intrinsic_reward:
                        phi_head_loss, metrics_data = self._compute_phi_loss(data=batch, metrics_data=metrics_data)
                        loss = phi_head_loss + loss
                    loss /= self.gradient_accumulation
                    with backward_ctx:
                        self.grad_scaler.scale(loss).backward()

                    metrics_data["actor/total_loss"] = loss.detach().item()
                    append_to_dict(metrics, metrics_data)

                torch.cuda.empty_cache()

                grad_norm, lr_list = self.optimizer_step()
                if self.use_intrinsic_reward:
                    self._soft_update_target_phi()
                data = {
                    "actor/grad_norm": grad_norm,
                    "actor/lr": lr_list[0],
                }
                if len(lr_list) > 1:
                    data["critic/lr"] = lr_list[1]
                append_to_dict(metrics, data)
        # put LR scheduler step here
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        clear_memory()
        mean_metric_dict = {key: np.mean(value) for key, value in metrics.items()}
        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )

        return mean_metric_dict

    def set_global_step(self, global_step) -> None:
        """
        Set the global step for the model, if needed.
        """
        if hasattr(self.model, "set_global_step"):
            self.model.set_global_step(global_step)


    def _compute_phi_loss(
        self,
        data: dict[str, torch.Tensor],
        metrics_data: dict[str, float],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        total_loss = 0.0
        phi_tau = self.intrinsic_cfg.phi_expectile
        phi_coef = self.intrinsic_cfg.phi_loss_coef
        gamma = self.cfg.algorithm.gamma
        s_not_goal = (data['step_to_goal']!=0.0).float()
        phi_dtype = self.model.phi_head.phi_list[0].fc1.weight.dtype
        phi_state_feats = data["phi_state_features"].to(dtype=phi_dtype).detach()
        phi_next_feats = data["phi_next_features"].to(dtype=phi_dtype).detach()
        phi_goal_feats = data["phi_goal_features"].to(dtype=phi_dtype).detach()
        phi_s = self.model.phi_head(phi_state_feats, return_phi=True)
        phi_goal = self.model.phi_head(phi_goal_feats, return_phi=True)
        with torch.no_grad():
            target_phi_s = self.model.target_phi_head(phi_state_feats, return_phi=True)
            target_phi_next = self.model.target_phi_head(phi_next_feats, return_phi=True)
            target_phi_goal = self.model.target_phi_head(phi_goal_feats, return_phi=True)
            v_sg_t = compute_v_sg(target_phi_s, target_phi_goal)
            v_next_t = compute_v_sg(target_phi_next, target_phi_goal)
            v_sg_mean = v_sg_t.mean(dim=1, keepdim=True)
            target_q = -s_not_goal + s_not_goal * gamma * v_next_t.min(dim=1, keepdim=True).values
            adv = target_q - v_sg_mean
        mask_phi = data["loss_mask"]
        mask_phi = mask_phi.any(dim=-1, keepdim=True).float()

        q = -s_not_goal + s_not_goal * gamma * v_next_t
        v_sg = compute_v_sg(phi_s, phi_goal)
        phi_loss = asymmetric_l2_loss(adv, q - v_sg, phi_tau)
        phi_loss = phi_loss * mask_phi if mask_phi is not None else phi_loss
        phi_loss = phi_loss.mean()
        total_loss = total_loss + phi_coef * phi_loss

        metrics_data.update({
            "phi/iql_loss": phi_loss.detach().item(),
            "phi/v_mean": v_sg.detach().mean().item(),
            "phi/v_max": v_sg.detach().max().item(),
            "phi/v_min": v_sg.detach().min().item(),
            "phi/q": q.detach().mean().item(),
            "phi/adv": adv.detach().mean().item(),
        })

        #contrasitive loss
        if "task_ids" in data:
            contrast_coef = self.intrinsic_cfg.get("contrastive_coef", 0.0)
            contrast_temp = self.intrinsic_cfg.get("contrastive_temp", 0.1)
            if contrast_coef > 0:
                task_ids = data["task_ids"].to(phi_goal.device)
                success_flags = data["episode_success"]
                success_flags = success_flags.any(dim=-1).to(phi_goal.device).bool() 
                goal_emb = phi_goal.mean(dim=1)
                goal_emb = F.normalize(goal_emb, p=2, dim=-1)
                sim_matrix = torch.matmul(goal_emb, goal_emb.transpose(0, 1)) / contrast_temp
                mask_pos = (task_ids.unsqueeze(1) == task_ids.unsqueeze(0)).float()
                success_mat = success_flags.unsqueeze(1) * success_flags.unsqueeze(0)
                mask_pos = mask_pos * success_mat.float()
                mask_pos.fill_diagonal_(0.0)
                exp_sim = torch.exp(sim_matrix)
                valid_mask = mask_pos.sum(dim=1) > 0
                self_sim = torch.exp(torch.diagonal(sim_matrix, 0))
                pos_sum = (exp_sim * mask_pos).sum(dim=1)
                numerator = torch.where(valid_mask, pos_sum, self_sim)
                neg_mask = 1.0 - mask_pos
                neg_mask.fill_diagonal_(0.0)
                neg_sum = (exp_sim * neg_mask).sum(dim=1)
                denominator = numerator + neg_sum + 1e-8
                loss_i = -torch.log(numerator / denominator)
                contrast_loss = loss_i.mean()
                total_loss = total_loss + contrast_coef * contrast_loss
                metrics_data["phi/contrast_loss"] = contrast_loss.detach().item()
                # writer.add_image('similarity_matrix', a.unsqueeze(0).unsqueeze(0), global_step=0)
                # metrics_data["contrast/mean_sim_all"] = sim_matrix.detach().mean().item()
        
        # supvervised loss
        if 'step_to_goal' in data:
            supv_coef = self.intrinsic_cfg.get("supervised_coef", 0.0)
            if supv_coef > 0:
                step_to_goal = data['step_to_goal'].to(phi_goal.device).to(v_sg.dtype)
                supv_loss = F.mse_loss(v_sg.mean(dim=-1, keepdim=True), -step_to_goal)
                total_loss = total_loss + supv_coef * supv_loss
                metrics_data["phi/supervised_loss"] = supv_loss.detach().item()

        metrics_data['phi/total_loss'] = total_loss.detach().item()
        return total_loss, metrics_data
