import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from rlinf.models import get_model


def reduce_phi_embeddings(phi: torch.Tensor, reduce_mode: str) -> torch.Tensor:
    if reduce_mode == "mean":
        return phi.mean(dim=1)
    if reduce_mode == "flatten":
        return phi.reshape(phi.shape[0], -1)
    raise ValueError(f"Unknown reduce mode: {reduce_mode}")


def pca_fit(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = embeddings.mean(axis=0, keepdims=True)
    centered = embeddings - mean
    _, _, v_t = np.linalg.svd(centered, full_matrices=False)
    components = v_t[:2].T
    return mean, components


def pca_transform(embeddings: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    return (embeddings - mean) @ components


def tsne_2d(embeddings: np.ndarray, seed: int, perplexity: float = 30.0) -> np.ndarray:
    try:
        from sklearn.manifold import TSNE
    except ImportError as exc:
        raise ImportError("t-SNE requires scikit-learn; install it or use --method pca.") from exc
    n_samples = embeddings.shape[0]
    if n_samples < 2:
        raise ValueError("t-SNE needs at least 2 samples.")
    max_perp = max(1.0, min(perplexity, float(n_samples - 1)))
    if max_perp != perplexity:
        print(f"Adjusted t-SNE perplexity from {perplexity} to {max_perp} for n={n_samples}.")
    tsne = TSNE(
        n_components=2,
        random_state=seed,
        perplexity=max_perp,
        init="pca",
        learning_rate="auto",
    )
    return tsne.fit_transform(embeddings)


class PrecomputedFeatureDataset(Dataset):
    """Loads goal embeddings inputs from precompute_encode_hidden_state.py output."""

    def __init__(self, feature_path: str | Path):
        feature_path = Path(feature_path)
        if not feature_path.is_file():
            raise FileNotFoundError(f"feature file not found: {feature_path}")
        payload = torch.load(feature_path, map_location="cpu")
        self.goal = payload["goal_state"]
        self.task_ids = payload.get("task_ids", None)
        meta = payload.get("meta", {})
        self.instruction_to_id: Optional[Dict[str, int]] = meta.get("instruction_to_id", None)

    def __len__(self) -> int:
        return self.goal.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = {"goal_feat": self.goal[idx]}
        if self.task_ids is not None:
            sample["task_id"] = self.task_ids[idx]
        return sample


def load_phi_heads(model, ckpt_path: Optional[str]):
    if ckpt_path is None:
        return
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "phi_head" in ckpt:
        model.phi_head.load_state_dict(ckpt["phi_head"])
    if "target_phi_head" in ckpt and hasattr(model, "target_phi_head"):
        model.target_phi_head.load_state_dict(ckpt["target_phi_head"])


def load_goal_pool_embeddings(ckpt_path: str, reduce_mode: str) -> tuple[np.ndarray, np.ndarray]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    goal_pool = ckpt.get("goal_ema_by_task", None)
    if goal_pool:
        items = sorted(goal_pool.items(), key=lambda kv: kv[0])
        goal_task_ids = np.array([int(k) for k, _ in items], dtype=np.int64)
        goal_embeds = torch.stack([torch.as_tensor(v) for _, v in items], dim=0)
    else:
        goal_embeds = ckpt.get("goal_phi_table", None)
        if goal_embeds is None:
            raise ValueError(f"No goal_ema_by_task or goal_phi_table found in checkpoint: {ckpt_path}")
        goal_embeds = torch.as_tensor(goal_embeds)
        goal_task_ids = np.arange(goal_embeds.shape[0], dtype=np.int64)
    goal_embeds = reduce_phi_embeddings(goal_embeds, reduce_mode)
    return goal_embeds.detach().float().cpu().numpy(), goal_task_ids


def collect_goal_embeddings_from_features(
    model,
    feature_path: Path,
    batch_size: int,
    reduce_mode: str,
    use_target_phi: bool,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, Optional[Dict[int, str]]]:
    dataset = PrecomputedFeatureDataset(feature_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    phi_dtype = next(model.phi_head.parameters()).dtype

    embeds: List[np.ndarray] = []
    task_ids: List[int] = []
    id_to_instr = None
    if dataset.instruction_to_id is not None:
        id_to_instr = {v: k for k, v in dataset.instruction_to_id.items()}

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=phi_dtype if device.type == "cuda" else None):
        for batch in dataloader:
            goal_feat = batch["goal_feat"].to(device=device, dtype=phi_dtype, non_blocking=True)
            tids = batch.get("task_id", None)
            if tids is not None:
                tids = tids.to(device)
            else:
                tids = torch.zeros(goal_feat.size(0), dtype=torch.long, device=device)

            phi_goal = model.phi_head.get_both_phi(goal_feat)
            phi_target = model.target_phi_head.get_both_phi(goal_feat) if use_target_phi and hasattr(model, "target_phi_head") else phi_goal
            phi = phi_target if use_target_phi else phi_goal
            phi = reduce_phi_embeddings(phi, reduce_mode)
            phi_np = phi.detach().float().cpu().numpy()

            embeds.extend(list(phi_np))
            task_ids.extend(tids.detach().cpu().tolist())

    return np.stack(embeds, axis=0), np.array(task_ids, dtype=np.int64), id_to_instr

import hydra
@hydra.main(
    version_base="1.1", config_path="../config", config_name="libero_10_offline_openvlaoft"
)
def main(cfg) -> None:
    feature_path = f'{cfg.data.path[0]}encode_hidden_state.pt'
    phi_checkpoint = cfg.reward.intrinsic_reward.phi_checkpoint
    batch_size = 256
    output_figure = 'phi_traj_means.png'
    goal_embeding_path = 'goal_embeds.pt'
    save_summary = 'phi_traj_means.npz'
    embedding = 'goal_embedding.pt'
    method = 'pca'  # or 'tsne'
    reduce = 'mean'  # or 'flatten'
    use_target_phi = False
    output = 'goal_phi_plot.png'
    plot_goal_pool = True
    pool_checkpoint = None  # if None, use phi_checkpoint    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = get_model(cfg.actor.model)
    model.to(device)
    load_phi_heads(model, phi_checkpoint)
    model.eval()

    # collect embeddings
    goal_embeds, goal_task_ids, id_to_instr = collect_goal_embeddings_from_features(
        model=model,
        feature_path=feature_path,
        batch_size=batch_size,
        reduce_mode=reduce,
        use_target_phi=use_target_phi,
        device=device,
    )
    torch.save(goal_embeds, goal_embeding_path)
    if method == "pca":
        mean, components = pca_fit(goal_embeds)
        coords = pca_transform(goal_embeds, mean, components)
    else:
        coords = tsne_2d(goal_embeds)
        mean = components = None

    pool_coords = pool_task_ids = None
    if plot_goal_pool:
        pool_embeds, pool_task_ids = load_goal_pool_embeddings(phi_checkpoint, reduce)
        if method == "pca":
            pool_coords = pca_transform(pool_embeds, mean, components)
        else:
            pool_coords = tsne_2d(pool_embeds)

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=goal_task_ids,
        cmap="tab20",
        s=18,
        alpha=0.25,  # make state points lighter
        label="goal states",
    )
    if pool_coords is not None:
        ax.scatter(
            pool_coords[:, 0],
            pool_coords[:, 1],
            c=pool_task_ids,
            cmap="tab20",
            s=80,
            alpha=0.95,  # keep goal pool vivid
            edgecolors="black",
            linewidths=0.4,
            zorder=3,
            label="goal pool",
        )
    ax.set_title(f"Goal phi embeddings ({method.upper()})")
    ax.set_xlabel("dim-1")
    ax.set_ylabel("dim-2")
    ax.grid(True, alpha=0.2)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("task_id")
    ax.legend(fontsize="small", loc="best")
    out_path = output or f"goal_phi_{method}.png"
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved goal embedding plot to {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
