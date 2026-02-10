import argparse
import os
import random
from pathlib import Path
from typing import Dict, List, Optional
from contextlib import nullcontext

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from rlinf.models import get_model


def compute_v_sg(phi_state: torch.Tensor, phi_goal: torch.Tensor) -> torch.Tensor:
    """Compute -||phi(s)-phi(g)||; returns [B, head, 1]."""
    dist = (phi_state - phi_goal).pow(2).sum(dim=-1, keepdim=True)
    return -torch.sqrt(dist.clamp(min=1e-6))


class PrecomputedFeatureDataset(Dataset):
    """Load features produced by examples/sft/precompute_encode_hidden_state.py."""

    def __init__(self, feature_path: str | Path):
        feature_path = Path(feature_path)
        if not feature_path.is_file():
            raise FileNotFoundError(f"feature file not found: {feature_path}")
        payload = torch.load(feature_path, map_location="cpu")
        required = ["state", "goal_state"]
        for k in required:
            if k not in payload:
                raise KeyError(f"'{k}' missing in {feature_path}")

        self.state = payload["state"]
        self.goal = payload["goal_state"]
        self.task_ids = payload.get("task_ids", None)
        self.is_terminal = payload.get("is_terminal", None)
        meta = payload.get("meta", {})
        self.instruction_to_id: Optional[Dict[str, int]] = meta.get("instruction_to_id", None)

    def __len__(self) -> int:
        return self.state.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = {
            "state_feat": self.state[idx],
            "goal_feat": self.goal[idx],
        }
        if self.task_ids is not None:
            sample["task_id"] = self.task_ids[idx]
        if self.is_terminal is not None:
            sample["is_terminal"] = self.is_terminal[idx]
        return sample


def load_phi_heads(model, ckpt_path: Optional[str]):
    """Load phi_head/target_phi_head from checkpoint produced by training."""
    if ckpt_path is None:
        return
    ckpt_path = str(ckpt_path)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    if "phi_head" in ckpt:
        model.phi_head.load_state_dict(ckpt["phi_head"])
    if "target_phi_head" in ckpt and hasattr(model, "target_phi_head"):
        model.target_phi_head.load_state_dict(ckpt["target_phi_head"])


def evaluate(cfg, feature_path: Path, phi_ckpt: Optional[str], batch_size: int, device: torch.device, max_trajs_per_task: int = 5):
    # Data
    dataset = PrecomputedFeatureDataset(feature_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    id_to_instr = None
    if dataset.instruction_to_id is not None:
        id_to_instr = {v: k for k, v in dataset.instruction_to_id.items()}


    # Model
    model = get_model(cfg.actor.model)
    model.to(device)
    load_phi_heads(model, phi_ckpt)
    model.eval()
    phi_dtype = next(model.phi_head.parameters()).dtype


    per_task_values: Dict[int, List[float]] = {}
    per_task_trajs: Dict[int, List[List[float]]] = {}
    current_traj: Dict[int, List[float]] = {}
    autocast_ctx = (
        torch.cuda.amp.autocast(dtype=phi_dtype)
        if device.type == "cuda" and phi_dtype in (torch.float16, torch.bfloat16)
        else torch.autocast("cpu", dtype=phi_dtype)
        if device.type == "cpu" and phi_dtype in (torch.float16, torch.bfloat16)
        else nullcontext()
    )
    with torch.no_grad(), autocast_ctx:
        for batch in dataloader:
            state_feat = batch["state_feat"].to(device=device, dtype=phi_dtype, non_blocking=True)
            goal_feat = batch["goal_feat"].to(device=device, dtype=phi_dtype, non_blocking=True)
            task_ids = batch.get("task_id", None)
            if task_ids is not None:
                task_ids = task_ids.to(device)
            else:
                # single dummy id if not provided
                task_ids = torch.zeros(state_feat.size(0), dtype=torch.long, device=device)
            term_flags = batch.get("is_terminal", None)
            if term_flags is not None:
                term_flags = term_flags.to(device)
            else:
                term_flags = torch.zeros_like(task_ids, dtype=torch.bool)

            phi_s = model.phi_head.get_both_phi(state_feat)  # [B, head, dim]
            phi_g = model.phi_head.get_both_phi(goal_feat)
            v_sg = compute_v_sg(phi_s, phi_g).mean(dim=1).squeeze(-1)  # [B]

            vals_cpu = v_sg.detach().cpu().tolist()
            tids_cpu = task_ids.detach().cpu().tolist()
            terms_cpu = term_flags.detach().cpu().tolist()
            for val, tid, term in zip(vals_cpu, tids_cpu, terms_cpu):
                tid = int(tid)
                per_task_values.setdefault(tid, []).append(val)
                current_traj.setdefault(tid, []).append(val)
                if term:
                    per_task_trajs.setdefault(tid, []).append(current_traj[tid])
                    current_traj[tid] = []
            # early break if all tasks reached quota
            if all(len(trajs) >= max_trajs_per_task for trajs in per_task_trajs.values()):
                break

    # summarize
    summary = {tid: float(np.mean(vals)) for tid, vals in per_task_values.items()}
    return summary, per_task_values, per_task_trajs, id_to_instr


def plot_summary(per_task_trajs: Dict[int, List[List[float]]], id_to_instr: Optional[Dict[int, str]], output_path: str):
    tasks = sorted(per_task_trajs.keys())
    n = len(tasks)
    fig, axes = plt.subplots(n, 1, figsize=(8, max(3, 2.5 * n)), sharex=False)
    if n == 1:
        axes = [axes]
    for ax, task in zip(axes, tasks):
        label = id_to_instr.get(task, f"task_{task}") if id_to_instr else f"task_{task}"
        trajs = per_task_trajs[task][:50]
        for idx, traj in enumerate(trajs):
            ax.plot(range(len(traj)), traj, linewidth=0.3, label=f"traj{idx+1}")
        ax.set_title(label)
        ax.set_xlabel("t")
        ax.set_ylabel("v(s,g)")
        ax.grid(True, alpha=0.3)
        if len(trajs) <= 5:
            ax.legend(fontsize="small")
    plt.tight_layout()
    out_dir = os.path.dirname(output_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output_path)
    print(f"[ok] saved figure to {output_path}")

import hydra
@hydra.main(
    version_base="1.1", config_path="../config", config_name="libero_10_offline_openvlaoft"
)
def main(cfg)->None:
    feature_path = f'{cfg.data.path[0]}encode_hidden_state.pt'
    phi_checkpoint = cfg.reward.intrinsic_reward.phi_checkpoint
    batch_size = 256
    output_figure = 'phi_traj_means.png'
    save_summary = 'phi_traj_means.npz'
    save_trajs = 'phi_per_task_trajs.pt'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    summary, per_task_values, per_task_trajs, id_to_instr = evaluate(
        cfg=cfg,
        feature_path=feature_path,
        phi_ckpt=phi_checkpoint,
        batch_size=batch_size,
        device=device,
        max_trajs_per_task=5,
    )


    if output_figure:
        plot_summary(per_task_trajs, id_to_instr, output_figure)

    if save_summary:
        out_path = Path(save_summary)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_path, summary=summary, per_task_values=per_task_values)
        print(f"[ok] saved summary to {out_path}")

    if save_trajs:
        traj_path = Path(save_trajs)
        traj_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(per_task_trajs, traj_path)
        print(f"[ok] saved per_task_trajs to {traj_path}")

if __name__ == "__main__":
    main()
