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

"""Qwen2.5-VLA (0.5B) embodied policy loader."""

import torch
from omegaconf import DictConfig

from .qwen_vla_action_model import QwenVLAForRLPolicy


def get_model(cfg: DictConfig, torch_dtype=torch.bfloat16):
    model = QwenVLAForRLPolicy(
        model_path=cfg.model_path,
        action_dim=cfg.get("action_dim", 7),
        num_action_chunks=cfg.get("num_action_chunks", 1),
        add_value_head=cfg.get("add_value_head", False),
        torch_dtype=torch_dtype,
        precision=cfg.precision,
        trust_remote_code=cfg.get("trust_remote_code", True),
        max_prompt_length=cfg.get("max_prompt_length", 256),
    )

    model.to(torch_dtype)
    return model

