import argparse
import math
import os
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from rlinf.data.datasets.libero_dataset import load_rlds_dataloader
from rlinf.models import get_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def preprocess_obs(model, instructions: List[str], obs: dict, device, dtype):
    primary_image = obs["img"]
    if primary_image.ndim == 4:
        primary_image = primary_image.unsqueeze(1)  # [B,1,C,H,W]
    max_prompt_length = model.max_prompt_length
    inputs = model.input_processor(
        text=instructions,
        images={"images": primary_image},
        proprio_states=obs["robot_state"],
        padding="max_length",
        max_length=max_prompt_length,
    )
    pixel_values = inputs["pixel_values"]
    pixel_values = pixel_values.to(device=device, dtype=dtype)
    bsz, num_img, channels, height, width = pixel_values.shape
    pixel_values = pixel_values.view(bsz, num_img * channels, height, width)
    return {
        "input_ids": inputs["input_ids"].to(device=device, dtype=torch.long),
        "attention_mask": inputs["attention_mask"].to(device=device, dtype=torch.bool),
        "pixel_values": pixel_values,
    }



def precompute_dataset_features(
    dataset,
    output_path: Path,
    model,
    device: torch.device,
    model_dtype: torch.dtype,
    store_dtype: torch.dtype,
    batch_size: int,
    num_workers: int,
    overwrite: bool,
    dataset_sizes: Sequence[int],
    instruction_to_id=None,
) -> None:
    if output_path.exists() and not overwrite:
        print(f"[skip] {output_path} exists (use --overwrite to replace)")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    total = len(dataset)
    state_feats = next_feats = goal_feats = None
    task_ids = steps_to_goal = success_flags = is_terminal = None
    write_idx = 0

    amp_ctx = (
        torch.cuda.amp.autocast(dtype=model_dtype)
        if device.type == "cuda" and model_dtype in (torch.float16, torch.bfloat16)
        else nullcontext()
    )

    cnt = 0
    for batch in tqdm(dataloader, desc="precompute", total=math.ceil(total / batch_size)):
        instructions = [
            f"In: What action should the robot take to {t.lower()}?\nOut: "
            for t in batch["instruction"]
        ]
        obs_inputs = preprocess_obs(model, instructions, batch["state"], device, model_dtype)
        next_inputs = preprocess_obs(model, instructions, batch["next_state"], device, model_dtype)
        goal_inputs = preprocess_obs(model, instructions, batch["goal_state"], device, model_dtype)

        with torch.inference_mode(), amp_ctx:
            input_ids_cat = torch.cat(
                [obs_inputs["input_ids"], next_inputs["input_ids"], goal_inputs["input_ids"]],
                dim=0,
            )
            attn_mask_cat = torch.cat(
                [
                    obs_inputs["attention_mask"],
                    next_inputs["attention_mask"],
                    goal_inputs["attention_mask"],
                ],
                dim=0,
            )
            pixels_cat = torch.cat(
                [obs_inputs["pixel_values"], next_inputs["pixel_values"], goal_inputs["pixel_values"]],
                dim=0,
            )
            latents_cat = model.encode_features(input_ids_cat, attn_mask_cat, pixels_cat)
            latent_s, latent_next, latent_goal = torch.chunk(latents_cat, 3, dim=0)

        if state_feats is None:
            feat_dim = latent_s.shape[-1]
            state_feats = torch.empty((total, feat_dim), dtype=store_dtype)
            next_feats = torch.empty((total, feat_dim), dtype=store_dtype)
            goal_feats = torch.empty((total, feat_dim), dtype=store_dtype)
            # metadata buffers
            task_ids = torch.zeros((total,), dtype=torch.long)
            steps_to_goal = torch.zeros((total,), dtype=torch.float32)
            is_success = torch.zeros((total,), dtype=torch.bool)
            is_terminal = torch.zeros((total,), dtype=torch.bool)

        bsz = latent_s.shape[0]
        state_feats[write_idx : write_idx + bsz] = latent_s.to("cpu", dtype=store_dtype)
        next_feats[write_idx : write_idx + bsz] = latent_next.to("cpu", dtype=store_dtype)
        goal_feats[write_idx : write_idx + bsz] = latent_goal.to("cpu", dtype=store_dtype)
        task_ids[write_idx : write_idx + bsz] = batch["task_id"].to(torch.long).cpu()
        steps_to_goal[write_idx : write_idx + bsz] = batch["steps_to_goal"].to(torch.float32).cpu()
        is_terminal[write_idx : write_idx + bsz] = batch["is_terminal"].to(torch.bool).cpu()
        is_success[write_idx : write_idx + bsz] = batch["is_success"].to(torch.bool).cpu()
        write_idx += bsz


        if cnt % 100 == 0:
            payload = {
                "state": state_feats,
                "next_state": next_feats,
                "goal_state": goal_feats,
                "task_ids": task_ids,
                "steps_to_goal": steps_to_goal,
                "is_traj_success": is_success,
                "is_terminal": is_terminal,
                "meta": {
                    "num_samples": total,
                    "feature_dim": state_feats.shape[-1],
                    "dtype": str(store_dtype),
                    "dataset_sizes": list(dataset_sizes),
                    "instruction_to_id": instruction_to_id,
                },
            }
            torch.save(payload, output_path)
            print(f"[ok] saved {output_path}")
        cnt+=1
    if write_idx != total:
        raise RuntimeError(f"mismatched sample count: wrote {write_idx}, expected {total}")

    
    payload = {
        "state": state_feats,
        "next_state": next_feats,
        "goal_state": goal_feats,
        "task_ids": task_ids,
        "steps_to_goal": steps_to_goal,
        "is_traj_success": is_success,
        "is_terminal": is_terminal,
        "meta": {
            "num_samples": total,
            "feature_dim": state_feats.shape[-1],
            "dtype": str(store_dtype),
            "dataset_sizes": list(dataset_sizes),
            "instruction_to_id": instruction_to_id,
        },
    }
    torch.save(payload, output_path)
    print(f"[ok] saved {output_path}")


import hydra
@hydra.main(
    version_base="1.1", config_path="config", config_name="libero_10_offline_openvlaoft"
)
def main(cfg) -> None:
    split = 'train'
    output_name = 'encode_hidden_state.pt'
    seed = 1234
    overwrite = True
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load model
    model = get_model(cfg.actor.model)

    # load dataset
    data_cfg = cfg.data
    batch_size = 16
    num_workers = 4
    dataloader, _ = load_rlds_dataloader(data_cfg, split_name=split)
    combined = dataloader.dataset
    dataset_list = combined.datasets if hasattr(combined, "datasets") else [combined]
    dataset_sizes = [len(ds) for ds in dataset_list]
    model_dtype = getattr(model, "dtype", next(model.parameters()).dtype)
    store_dtype = model_dtype
    instruction_to_id = getattr(dataloader, "instruction_to_id", None)

    output_path = cfg.data.path[0] + output_name
    #todo: turn output_path str to os.Path
    output_path = Path(output_path)

    precompute_dataset_features(
        dataset=combined,
        output_path=output_path,
        model=model,
        device=device,
        model_dtype=model_dtype,
        store_dtype=store_dtype,
        batch_size=batch_size,
        num_workers=num_workers,
        overwrite=overwrite,
        dataset_sizes=dataset_sizes,
        instruction_to_id=instruction_to_id,
    )


if __name__ == "__main__":
    main()
