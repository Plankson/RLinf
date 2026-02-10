import math
import os
import random
from contextlib import nullcontext
from itertools import cycle
from pathlib import Path
from typing import Dict, Optional

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
import wandb

from rlinf.models import get_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _module_state_cpu(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in module.state_dict().items()}


def load_phi_heads(model, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.phi_head.load_state_dict(ckpt["phi_head"])
    model.target_phi_head.load_state_dict(ckpt["target_phi_head"])
    print(f"successfully load ckpt from {ckpt_path}")
    return ckpt.get("goal_phi_table", None), ckpt.get("instruction2id", None)


def soft_update_target_phi(model, tau: float) -> None:
    with torch.no_grad():
        for src, tgt in zip(model.phi_head.parameters(), model.target_phi_head.parameters()):
            tgt.data.mul_(1 - tau).add_(src.data, alpha=tau)


def asymmetric_l2_loss(u: torch.Tensor, diff: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.abs(tau - (u < 0).float()) * diff**2


def compute_phi_loss_from_latents(
    model,
    latent_s: torch.Tensor,
    latent_next: torch.Tensor,
    latent_goal: torch.Tensor,
    target_value: Optional[torch.Tensor] = None,
    task_ids: Optional[torch.Tensor] = None,
    success_flags: Optional[torch.Tensor] = None,
    is_terminal: Optional[torch.Tensor] = None,
    gamma: float = 0.99,
    tau: float = 0.7,
    phi_coef: float = 1.0,
    contrastive_coef: float = 0.0,
    contrastive_temp: float = 0.1,
    sup_coef: float = 0.3,
):
    s_not_goal = (~is_terminal).to(dtype=latent_s.dtype).unsqueeze(-1)
    with torch.no_grad():
        target_phi_s = model.target_phi_head.get_both_phi(latent_s)
        target_phi_next = model.target_phi_head.get_both_phi(latent_next)
        target_phi_goal = model.target_phi_head.get_both_phi(latent_goal)

    phi_s = model.phi_head.get_both_phi(latent_s)
    phi_goal = model.phi_head.get_both_phi(latent_goal)

    def v_sg(phi_state, phi_goal_state):
        dist = (phi_state - phi_goal_state).pow(2).sum(dim=-1, keepdim=True)
        return -torch.sqrt(dist.clamp(min=1e-6))

    # import pdb; pdb.set_trace()
    v_sg_t = v_sg(target_phi_s, target_phi_goal)
    v_next_t = v_sg(target_phi_next, target_phi_goal)
    v_sg_mean = v_sg_t.mean(dim=1, keepdim=True)
    target_q = -s_not_goal.unsqueeze(-1) + s_not_goal.unsqueeze(-1) * gamma * v_next_t.min(dim=1, keepdim=True).values
    adv = target_q - v_sg_mean

    q = -s_not_goal.unsqueeze(-1) + s_not_goal.unsqueeze(-1) * gamma * v_next_t
    v_sg_pred = v_sg(phi_s, phi_goal)
    phi_loss = asymmetric_l2_loss(adv, q - v_sg_pred, tau)
    phi_loss = phi_coef * phi_loss.mean()

    log = {"train/phi_loss": phi_loss.item()}
    total_loss = phi_loss
    contrast_loss = None

    if contrastive_coef and contrastive_coef > 0 :
        goal_emb = phi_goal.mean(dim=1)
        goal_emb = torch.nn.functional.normalize(goal_emb, p=2, dim=-1)
        sim_matrix = goal_emb @ goal_emb.T / contrastive_temp

        task_ids = task_ids.to(goal_emb.device)
        mask_pos = (task_ids.unsqueeze(1) == task_ids.unsqueeze(0)).float()
        success_flags = success_flags.to(goal_emb.device).bool()
        success_mat = success_flags.unsqueeze(1) & success_flags.unsqueeze(0)
        mask_pos = mask_pos * success_mat.float()
        mask_pos.fill_diagonal_(0.0)

        exp_sim = torch.exp(sim_matrix)
        pos_sum = (exp_sim * mask_pos).sum(dim=1)
        pos_count = mask_pos.sum(dim=1)
        self_sim = torch.exp(torch.diagonal(sim_matrix, 0))
        numerator = torch.where(pos_count > 0, pos_sum, self_sim)

        neg_mask = 1.0 - mask_pos
        neg_mask.fill_diagonal_(0.0)
        neg_sum = (exp_sim * neg_mask).sum(dim=1)
        denominator = numerator + neg_sum + 1e-8

        loss_i = -torch.log(numerator / denominator)
        contrast_loss = loss_i.mean()
        total_loss = total_loss + contrastive_coef * contrast_loss
        log["train/contrastive_loss"] = contrast_loss.item()

    target_v = -target_value.to(device=v_sg_pred.device, dtype=v_sg_pred.dtype)
    target_v = target_v.unsqueeze(1).unsqueeze(1)
    value_loss = F.mse_loss( v_sg_pred, target_v.expand([-1, *v_sg_pred.shape[1:]]) )
    total_loss = total_loss + sup_coef * value_loss
    log["train/supervised_value_loss"] = value_loss.item()

    log["train/total_phi_loss"] = total_loss.item()
    return total_loss, adv.reshape(adv.shape[0], -1), log, phi_goal.detach()


class PrecomputedFeatureDataset(Dataset):
    """Loads pre-encoded hidden states from torch.save output of precompute_encode_hidden_state.py."""

    def __init__(self, feature_path):
        feature_path = Path(feature_path)
        payload = torch.load(feature_path, map_location="cpu")

        self.state = payload["state"]
        self.next_state = payload["next_state"]
        self.goal_state = payload["goal_state"]
        self.meta = payload.get("meta", {})

        self.steps_to_goal = payload['steps_to_goal']
        self.task_id = payload['task_ids']
        self.is_traj_success = payload['is_traj_success']
        self.is_terminal = payload['is_terminal']
        self.instruction_to_id = self.meta.get("instruction_to_id", None)

    def __len__(self) -> int:
        return self.state.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = {
            "state_feat": self.state[idx],
            "next_state_feat": self.next_state[idx],
            "goal_feat": self.goal_state[idx],
            "steps_to_goal": self.steps_to_goal[idx],
            "task_id": self.task_id[idx],
            "is_traj_success": self.is_traj_success[idx],
            "is_terminal": self.is_terminal[idx],
        }
        return sample


def save_checkpoint(model, step: int, save_dir: Path, goal_phi_table: torch.Tensor, instruction2id) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {"step": step}
    ckpt["phi_head"] = _module_state_cpu(model.phi_head)
    if hasattr(model, "target_phi_head"):
        ckpt["target_phi_head"] = _module_state_cpu(model.target_phi_head)
    ckpt["goal_phi_table"] = goal_phi_table.cpu()
    ckpt['instruction2id'] = instruction2id
    ckpt['step'] = step
    ckpt_step = save_dir / f"phi_heads_preencoded_step_{step}.pt"
    ckpt_latest = save_dir / "phi_heads_preencoded_latest.pt"
    torch.save(ckpt, ckpt_step)
    torch.save(ckpt, ckpt_latest)
    return ckpt_latest


@hydra.main(
    version_base="1.1", config_path="config", config_name="libero_10_offline_openvlaoft"
)
def main(cfg) -> None:
    # Basic setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = cfg.get("seed", 1234)
    set_seed(seed)

    # Resolve feature file
    default_feature_name = getattr(cfg, "feature_filename", "encode_hidden_state.pt")
    default_feature_path = Path(cfg.data.path[0]) / default_feature_name
    feature_path = Path(getattr(cfg, "feature_path", default_feature_path))

    # Model
    model = get_model(cfg.actor.model)
    model.to(device)
    # Only train phi_head; freeze everything else.
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("phi_head")
    phi_params = [p for p in model.parameters() if p.requires_grad]

    phi_cfg = cfg.reward.intrinsic_reward
    phi_dtype = next(model.phi_head.parameters()).dtype

    # Data
    dataset = PrecomputedFeatureDataset(feature_path)
    batch_size = cfg.data.get("batch_size", 64)
    num_workers = cfg.get("num_workers", 0)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    data_iter = cycle(dataloader)
    loaded_goal_table, loaded_instr2id = load_phi_heads(
        model,
        '/data/users/qingyunpeng/model/openvla-oft/Openvla-oft-SFT-libero10-traj1/phi_heads_preencoded_latest.pt'
    )
    # Optimizer / scheduler
    optim_cfg = cfg.actor.optim
    lr = optim_cfg.get("offline_lr", 3e-4)
    betas = (
        optim_cfg.get("adam_beta1", 0.9),
        optim_cfg.get("adam_beta2", 0.999),
    )
    adam_eps = optim_cfg.get("adam_eps", 1e-8)
    optimizer = torch.optim.AdamW(phi_params, lr=lr, betas=betas, eps=adam_eps)


    # Hyperparameters
    gamma = cfg.algorithm.get("gamma", 0.99)
    phi_loss_coef = phi_cfg.get("phi_loss_coef", 1.0)
    phi_expectile = phi_cfg.get("phi_expectile", 0.7)
    contrastive_coef = phi_cfg.get("contrastive_coef", 0.0)
    contrastive_temp = phi_cfg.get("contrastive_temp", 0.1)
    phi_target_tau = phi_cfg.get("phi_target_tau", 0.005)
    sup_coef = phi_cfg.get('supervised_coef', 0.3)
    goal_ema_decay = phi_cfg.get("goal_ema_decay", 0.99)

    max_steps = 20000
    save_every = 1000

    wandb.init(project='rlinf', name=cfg.runner.logger.experiment_name, config=OmegaConf.to_container(cfg, resolve=True))

    save_dir = cfg.actor.model.model_path
    model.train()

    instruction2id = loaded_instr2id if loaded_instr2id is not None else dataset.instruction_to_id
    num_tasks = len(instruction2id)
    # goal_phi_table: shape [task_num, 2, embed_dim]
    goal_phi_table = loaded_goal_table.clone().to(torch.float32) if loaded_goal_table is not None else None
    import tqdm
    for step in tqdm.tqdm(range(max_steps)):
        batch = next(data_iter)
        optimizer.zero_grad()

        state_feat = batch["state_feat"].to(device=device, dtype=phi_dtype, non_blocking=True)
        next_feat = batch["next_state_feat"].to(device=device, dtype=phi_dtype, non_blocking=True)
        goal_feat = batch["goal_feat"].to(device=device, dtype=phi_dtype, non_blocking=True)
        task_id = batch["task_id"].to(device)
        success_flags = batch['is_traj_success'].to(device)
        is_terminal = batch['is_terminal'].to(device)
        target_value = batch['steps_to_goal'].to(device)

        amp_ctx = (
            torch.cuda.amp.autocast(dtype=phi_dtype)
            if phi_dtype in (torch.float16, torch.bfloat16) and device.type == "cuda"
            else nullcontext()
        )

        with amp_ctx:
            loss, adv, log, goal_phi = compute_phi_loss_from_latents(
                model=model,
                latent_s=state_feat,
                latent_next=next_feat,
                latent_goal=goal_feat,
                target_value=target_value,
                task_ids=task_id,
                success_flags=success_flags,
                is_terminal=is_terminal,
                gamma=gamma,
                tau=phi_expectile,
                phi_coef=phi_loss_coef,
                contrastive_coef=contrastive_coef,
                contrastive_temp=contrastive_temp,
                sup_coef=sup_coef,
            )

        loss.backward()
        clip_val = getattr(cfg.actor.optim, "clip_grad", None)
        clip_grad_norm_(model.phi_head.parameters(), 1.0)
        optimizer.step()
        soft_update_target_phi(model, phi_target_tau)
        # update per-task soft average
        # lazy init goal_phi_table with correct shape [task_num, head(=2), embed_dim]
        if goal_phi_table is None or goal_phi_table.shape[1:] != goal_phi.shape[1:]:
            goal_phi_table = torch.zeros((num_tasks, *goal_phi.shape[1:]), dtype=torch.float32)

        goal_vec = goal_phi.detach().to("cpu", dtype=torch.float32)
        task_id_cpu = task_id.detach().to("cpu")
        for ind in range(goal_vec.shape[0]):
            idx = int(task_id_cpu[ind].item())
            feat = goal_vec[ind]
            goal_phi_table[idx] = goal_ema_decay * goal_phi_table[idx] + (1 - goal_ema_decay) * feat

        # wandb logging
        log_payload = {"train/lr": optimizer.param_groups[0]["lr"], **log}
        wandb.log(log_payload, step=step)

        if step % save_every == 0 or (step + 1) == max_steps:
            ckpt_path = save_checkpoint(model, step + 1, Path(save_dir), goal_phi_table, instruction2id)
            print(f"[step {step+1}] saved {ckpt_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
