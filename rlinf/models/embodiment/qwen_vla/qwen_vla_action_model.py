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

"""Minimal Qwen2.5-VL (0.5B) → action policy wrapper for RLinf embodied stack.

This wraps the official Qwen2.5-VL HuggingFace checkpoint as a VLA that maps
{images, text prompt, proprio} → continuous actions. It mirrors the OpenVLA
interface so rollout/actor code stays unchanged.

Assumptions:
- HF model is a causal LM with vision projector (Qwen2.5-VL 0.5B) whose tokenizer
  uses BOS=1 and padding_side="left".
- Actions are represented via appended special tokens that we interpret as
  per-dimension logits (same trick as OpenVLA: discrete bin centers then
  unnormalize with dataset stats stored beside the checkpoint).
- A small MLP value head can be toggled for PPO.

You can further adapt the action head if your checkpoint encodes actions
differently; keep the public `predict_action_batch` signature stable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import copy
import json
import os
import numpy as np
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.modules.resnet_mlp import SkillHead
from rlinf.models.embodiment.modules.value_head import ValueHead
from rlinf.utils.utils import compute_logprobs_from_logits


@dataclass
class ActionNormStats:
    q01: np.ndarray
    q99: np.ndarray
    mask: np.ndarray


class QwenVLAForRLPolicy(BasePolicy, torch.nn.Module):
    def __init__(
        self,
        model_path: str,
        action_dim: int,
        num_action_chunks: int,
        add_value_head: bool,
        torch_dtype: torch.dtype,
        precision: str,
        trust_remote_code: bool,
        max_prompt_length: int,
        add_progress_heads: bool = False,
        progress_num_blocks: int = 2,
        progress_hidden_dim: int = 32,
        progress_output_dim: int = 4,
        progress_head_num: int = 2,
    ):
        torch.nn.Module.__init__(self)
        BasePolicy.__init__(self)
        self.model_path = model_path
        self.action_dim = action_dim
        self.num_action_chunks = num_action_chunks
        self.max_prompt_length = max_prompt_length

        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True,
        )

        hidden_size = self.model.config.hidden_size
        self.vocab_size = self.model.config.vocab_size

        # Optional value head for PPO
        self.value_head = None
        if add_value_head:
            self.value_head = ValueHead(
                input_dim=hidden_size,
                hidden_sizes=(hidden_size, hidden_size // 2, 256),
                output_dim=1,
                activation="gelu",
                bias_last=True,
            )

        # Optional phi heads for intrinsic reward (IQL-style)
        self.add_progress_heads = add_progress_heads
        if self.add_progress_heads:
            self.phi_head = SkillHead(
                input_dim=hidden_size,
                hidden_dim=progress_hidden_dim,
                output_dim=progress_output_dim,
                num_blocks=progress_num_blocks,
                head_num=progress_head_num,
            )
            self.target_phi_head = copy.deepcopy(self.phi_head)

        # Load action normalization stats if present
        stats_path = os.path.join(model_path, "dataset_statistics.json")
        if os.path.isfile(stats_path):
            with open(stats_path, "r") as f:
                stats = json.load(f)
            action_stats = stats[list(stats.keys())[0]]["action"]
            self.action_stats = ActionNormStats(
                q01=np.array(action_stats["q01"]),
                q99=np.array(action_stats["q99"]),
                mask=np.array(action_stats.get("mask", np.ones_like(action_stats["q01"], dtype=bool))),
            )
        else:
            self.action_stats = None

    # === BasePolicy interface ===
    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        elif forward_type == ForwardType.SFT:
            return self.sft_forward(**kwargs)
        raise NotImplementedError

    def default_forward(
        self,
        forward_inputs: Optional[dict[str, torch.Tensor]] = None,
        compute_logprobs: bool = False,
        compute_entropy: bool = False,
        compute_values: bool = False,
        **generate_kwargs,
    ):
        # forward_inputs prepared during rollout training time
        input_ids = forward_inputs["input_ids"]
        attention_mask = forward_inputs["attention_mask"]
        pixel_values = forward_inputs["pixel_values"]
        action_tokens = forward_inputs["action_tokens"]

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=action_tokens,
            output_hidden_states=compute_values,
        )

        # Match OpenVLA-OFT: use the token right before the generated action block
        # as a compact trajectory summary for value/phi heads. Fall back to last token
        # if sequence is short (should not happen in normal use).
        value_feat = None
        phi_feat = None
        if compute_values and outputs.hidden_states:
            hidden_states = outputs.hidden_states[-1]  # [B, seq, hidden]
            feat_idx = hidden_states.shape[1] - self.action_dim * self.num_action_chunks - 1
            feat_idx = feat_idx if feat_idx >= 0 else -1
            value_feat = hidden_states[:, feat_idx, :]
            phi_feat = value_feat

        result = {"logprobs": None, "entropy": None}
        if compute_logprobs:
            logits = outputs.logits  # [B, L, vocab]
            # Only last action_dim*num_chunks tokens were labels
            logits = logits[:, -self.action_dim * self.num_action_chunks :, :]
            logprobs = compute_logprobs_from_logits(
                logits=logits.reshape(-1, self.vocab_size),
                target=action_tokens.reshape(-1),
            ).reshape(logits.shape[0], self.num_action_chunks, self.action_dim)
            result["logprobs"] = logprobs.float()

        if compute_entropy and compute_logprobs:
            # simple entropy from logits over action bins
            logits = outputs.logits[:, -self.action_dim * self.num_action_chunks :, :]
            probs = logits.softmax(dim=-1)
            entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
            entropy = entropy.reshape(logits.shape[0], self.num_action_chunks, self.action_dim)
            result["entropy"] = entropy.float()

        if compute_values and self.value_head is not None:
            values = self.value_head(value_feat)
            result["values"] = values

        if self.add_progress_heads and compute_values:
            result["phi_list_embed"] = self.phi_head(phi_feat, return_phi=True)
            result["target_phi_list_embed"] = self.target_phi_head(phi_feat, return_phi=True)

        return result

    # === SFT path (offline BC) ===
    def sft_forward(self, data: dict[str, torch.Tensor], **kwargs):
        """Supervised fine-tune with ground-truth continuous actions.

        Args:
            data: {"observation": {...}, "actions": Tensor [B, chunk, action_dim]}
        Returns:
            loss tensor scalar
        """
        obs = data["observation"]
        actions = data["actions"]  # [B, chunk, action_dim]

        # Build task prompts
        task_descs = [f"In: What action should the robot take to {t.lower()}?\nOut: " for t in obs["instruction"]]

        # Images: main camera only; processor can handle Tensor [B,1,C,H,W]
        images = obs["state"]["img"]  # [B, C, H, W]
        if images.ndim == 4:
            images = images.unsqueeze(1)  # [B,1,C,H,W]

        processor_inputs = self.processor(
            text=task_descs,
            images=images,
            padding="max_length",
            max_length=self.max_prompt_length,
            return_tensors="pt",
        )
        processor_inputs = {k: v.to(self.model.device) for k, v in processor_inputs.items()}

        # Convert continuous actions -> tokens (same discretization as predict)
        action_tokens = self._actions_to_tokens(actions)  # [B, chunk, action_dim]
        action_tokens = action_tokens.reshape(actions.shape[0], -1).to(self.model.device)

        forward_inputs = {
            "input_ids": processor_inputs["input_ids"],
            "attention_mask": processor_inputs["attention_mask"],
            "pixel_values": processor_inputs["pixel_values"],
            "action_tokens": action_tokens,
        }

        result = self.default_forward(
            forward_inputs=forward_inputs,
            compute_logprobs=True,
            compute_entropy=False,
            compute_values=False,
        )

        logprobs = result["logprobs"]  # [B, chunk, action_dim]
        loss = -logprobs.mean()
        return loss

    # === Helpers ===
    def _normalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Normalize continuous actions to [-1,1] using stored stats if available."""
        if self.action_stats is None:
            # assume already in [-1,1]
            return actions.clamp(-1.0, 1.0)

        high = torch.as_tensor(self.action_stats.q99, device=actions.device, dtype=actions.dtype)
        low = torch.as_tensor(self.action_stats.q01, device=actions.device, dtype=actions.dtype)
        mask = torch.as_tensor(self.action_stats.mask, device=actions.device, dtype=torch.bool)

        repeat = actions.shape[-1] // high.shape[0]
        high = high.repeat(repeat)
        low = low.repeat(repeat)
        mask = mask.repeat(repeat)

        normalized = torch.where(
            mask,
            2 * (actions - low) / (high - low + 1e-8) - 1,
            actions,
        )
        return normalized.clamp(-1.0, 1.0)

    def _actions_to_tokens(self, actions: torch.Tensor) -> torch.LongTensor:
        """Convert continuous actions to discrete token ids (inverse of predict mapping)."""
        normalized = self._normalize_actions(actions)  # [B, chunk, action_dim]
        bin_centers = torch.linspace(-1, 1, self.vocab_size, device=actions.device)
        # find nearest bin
        flat = normalized.reshape(-1).unsqueeze(-1)  # [B*chunk*dim, 1]
        # compute abs diff to centers efficiently
        dists = (flat - bin_centers).abs()
        idx = dists.argmin(dim=-1)  # [B*chunk*dim]
        token_ids = self.vocab_size - (idx + 1)  # align with predict mapping
        return token_ids.reshape(actions.shape[0], actions.shape[1], actions.shape[2])

    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: int = -1,
        top_p: float = 1.0,
        max_new_tokens: int | None = None,
        **kwargs,
    ):
        """Generate actions given environment observations.

        Returns:
            actions: np.ndarray [B, num_chunks, action_dim]
            result: dict with prev_logprobs, prev_values, forward_inputs
        """
        task_descs = [
            f"In: What action should the robot take to {t.lower()}?\nOut: "
            for t in env_obs["task_descriptions"]
        ]

        images = env_obs["main_images"]
        if images.ndim == 4:
            images = images.unsqueeze(1)  # [B,1,H,W,C]
        images = images.permute(0, 1, 4, 2, 3)  # to [B,1,C,H,W]

        processor_inputs = self.processor(
            text=task_descs,
            images=images,
            padding="max_length",
            max_length=self.max_prompt_length,
            return_tensors="pt",
        )
        processor_inputs = {k: v.to(self.model.device) for k, v in processor_inputs.items()}

        gen_out = self.model.generate(
            **processor_inputs,
            do_sample=do_sample,
            temperature=temperature,
            top_k=None if top_k == -1 else top_k,
            top_p=top_p,
            max_new_tokens=self.action_dim * self.num_action_chunks,
            output_scores=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )

        action_token_ids = gen_out.sequences[:, -self.action_dim * self.num_action_chunks :]
        scores = torch.stack(gen_out.scores, dim=1)  # [B, action_dim*num_chunks, vocab]

        # Map token ids to discretized actions: assume last vocab bins are action bins
        # We clip to ensure inside bin range (same as OpenVLA style)
        discretized = self.vocab_size - action_token_ids.cpu().numpy()
        discretized = np.clip(discretized - 1, 0, scores.shape[-1] - 1)
        bin_centers = np.linspace(-1, 1, scores.shape[-1])
        normalized = np.asarray([bin_centers[d] for d in discretized])

        if self.action_stats is not None:
            mask = self.action_stats.mask
            high, low = self.action_stats.q99, self.action_stats.q01
            normalized_shape = normalized.shape[-1]
            repeat = normalized_shape // high.shape[0]
            mask = np.repeat(mask, repeat)
            high = np.repeat(high, repeat)
            low = np.repeat(low, repeat)
            actions = np.where(
                mask,
                0.5 * (normalized + 1) * (high - low + 1e-8) + low,
                normalized,
            )
        else:
            actions = normalized

        # logprobs per action token
        prev_logprobs = compute_logprobs_from_logits(
            logits=scores.reshape(-1, self.vocab_size),
            target=action_token_ids.reshape(-1),
        ).reshape(action_token_ids.shape[0], self.num_action_chunks, self.action_dim)

        # Use hidden state right before the generated action block as phi/value feature
        decoder_hs_last = gen_out.decoder_hidden_states[-1]  # [B, seq, hidden]
        feat_idx = decoder_hs_last.shape[1] - self.action_dim * self.num_action_chunks - 1
        feat_idx = feat_idx if feat_idx >= 0 else -1
        phi_features = decoder_hs_last[:, feat_idx, :]

        forward_inputs = {
            "input_ids": processor_inputs["input_ids"],
            "attention_mask": processor_inputs["attention_mask"],
            "pixel_values": processor_inputs["pixel_values"],
            "action_tokens": action_token_ids,
            "phi_features": phi_features,
        }

        if self.value_head is not None:
            prev_values = self.value_head(phi_features)
        else:
            prev_values = torch.zeros(
                action_token_ids.shape[0], self.num_action_chunks, 1, device=self.model.device, dtype=prev_logprobs.dtype
            )

        if self.add_progress_heads:
            # Expose phi embeddings for intrinsic reward pipeline
            phi_list_embed = self.phi_head(phi_features, return_phi=True)
            target_phi_list_embed = self.target_phi_head(phi_features, return_phi=True)
        else:
            phi_list_embed = None
            target_phi_list_embed = None

        result = {
            "prev_logprobs": prev_logprobs,
            "prev_values": prev_values,
            "forward_inputs": forward_inputs,
            "phi_list_embed": phi_list_embed,
            "target_phi_list_embed": target_phi_list_embed,
        }

        chunk_actions = actions.reshape(-1, self.num_action_chunks, self.action_dim)
        return chunk_actions, result

    # === Utility ===
    def set_global_step(self, global_step):
        # Placeholder to keep interface parity
        self.global_step = global_step
