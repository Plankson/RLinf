"""Plot reward curves for each task and merge them into one PDF.

This script reads two torch tensors saved in the current directory:
- `v.pt`: reward values with shape either (steps, trajectories) or
  (trajectories, steps).
- `task.pt`: task id for each trajectory, shape (trajectories,).

It creates a single PDF containing one page per task, with all trajectories of
that task plotted and labeled by trajectory id. Optionally it can also emit
per-task PNGs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import torch


def load_tensors(v_path: Path, task_path: Path):
    """Load tensors from disk onto CPU."""

    v = torch.load(v_path, map_location="cpu")
    task_ids = torch.load(task_path, map_location="cpu")

    if task_ids.dim() != 1:
        raise ValueError(f"task_ids should be 1-D, got shape {tuple(task_ids.shape)}")

    return v, task_ids


def align_rewards(v: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
    """Return rewards shaped (trajectories, steps).

    Accept input v of shape (steps, trajectories) or (trajectories, steps).
    """

    traj_count = task_ids.numel()

    if v.dim() != 2:
        raise ValueError(f"Expected 2-D rewards tensor, got {v.dim()}-D")

    if v.shape[0] == traj_count:
        # already (trajectories, steps)
        return v
    if v.shape[1] == traj_count:
        # currently (steps, trajectories)
        return v.T

    raise ValueError(
        "Rewards shape does not match task_ids. "
        f"v shape {tuple(v.shape)}, task_ids length {traj_count}"
    )


def _chunk(seq, size):
    """Yield consecutive chunks of a sequence with a maximum length."""

    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def plot_by_task(rewards: torch.Tensor, task_ids: torch.Tensor, out_dir: Path):
    """Create one PDF (and optional PNGs) with trajectory reward curves per task.

    For each task, trajectories are grouped in chunks of 4. Each chunk becomes
    one subplot (one page) containing up to 4 trajectories plotted together.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    unique_tasks = torch.unique(task_ids).tolist()

    steps = rewards.shape[1]
    x = list(range(steps))

    pdf_path = out_dir / "task_rewards.pdf"
    png_paths = []

    with PdfPages(pdf_path) as pdf:
        for task in unique_tasks:
            indices = (task_ids == task).nonzero(as_tuple=True)[0]
            traj_ids = indices.tolist()

            for chunk_id, chunk in enumerate(_chunk(traj_ids, 4), start=1):
                fig, ax = plt.subplots(figsize=(8, 5))
                for idx in chunk:
                    ax.plot(x, rewards[idx].tolist(), label=f"traj {idx}")

                ax.set_title(f"Task {task} trajectories {chunk_id}")
                ax.set_xlabel("timestep")
                ax.set_ylabel("reward")
                ax.legend(title="trajectory id", fontsize=9)
                fig.tight_layout()

                pdf.savefig(fig)

                # Keep individual PNGs for quick glance if desired.
                png_path = out_dir / f"task_{task}_chunk_{chunk_id}.png"
                fig.savefig(png_path, dpi=150)
                png_paths.append(png_path)

                plt.close(fig)

    return pdf_path, png_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v", default="v.pt", help="Path to reward tensor (default: v.pt)")
    parser.add_argument(
        "--tasks", default="task.pt", help="Path to task id tensor (default: task.pt)"
    )
    parser.add_argument("--out", default="plots", help="Directory to store outputs (default: plots)")
    args = parser.parse_args()

    v_path = Path(args.v)
    task_path = Path(args.tasks)
    out_dir = Path(args.out)

    v, task_ids = load_tensors(v_path, task_path)
    rewards = align_rewards(v, task_ids)
    pdf_path, png_paths = plot_by_task(rewards, task_ids, out_dir)

    print("Generated PDF:")
    print(f" - {pdf_path}")
    print("Also wrote PNGs:")
    for p in png_paths:
        print(f" - {p}")


if __name__ == "__main__":
    main()
