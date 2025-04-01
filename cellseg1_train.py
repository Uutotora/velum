import json
from pathlib import Path
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
import torch
import torch.optim as optim
import yaml
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from cell_loss import cell_prob_mse_loss, cross_entropy_loss
from data.dataset import TrainDataset
from gpu_memory_tracker import GPUMemoryTracker
from peft.sam_lora_image_encoder_mask_decoder import LoRA_Sam
from sampler import create_collate_fn
from segment_anything import sam_model_registry
from set_environment import set_env, DEVICE


def prepare_directories(config: Dict):
    Path(config["result_pth_path"]).parent.mkdir(exist_ok=True, parents=True)


def load_dataset(config: Dict) -> TrainDataset:
    return TrainDataset(
        image_dir=Path(config["train_image_dir"]),
        mask_dir=Path(config["train_mask_dir"]),
        resize_size=config["resize_size"],
        patch_size=config["patch_size"],
        train_id=config["train_id"],
        duplicate_data=config["duplicate_data"],
    )


def load_model(config: Dict) -> LoRA_Sam:
    model = sam_model_registry[config["vit_name"]](checkpoint=config["model_path"], image_size=config["sam_image_size"])
    return LoRA_Sam(model, config).to(DEVICE)


def setup_training(
    config: Dict, model: LoRA_Sam, train_dataset: TrainDataset
) -> Tuple[DataLoader, optim.Optimizer, OneCycleLR]:
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["base_lr"],
    )
    custom_collate_func = create_collate_fn(config)
    trainloader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True,
        collate_fn=custom_collate_func,
    )
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config["base_lr"],
        total_steps=config["epoch_max"]
        * (len(trainloader) + config["gradient_accumulation_step"] - 1)
        // config["gradient_accumulation_step"],
        pct_start=config["onecycle_lr_pct_start"],
    )
    return trainloader, optimizer, scheduler


def to_tensor(
    images: List[np.ndarray], all_points: List[List[np.ndarray]], image_size: int
) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
    tensor_images = [torch.as_tensor(image.transpose(2, 0, 1), dtype=torch.float).to(DEVICE) for image in images]
    items = [
        {
            "point_coords": torch.as_tensor(np.stack(points).astype(np.int64), dtype=torch.float)[:, None, :].to(DEVICE),
            "point_labels": torch.ones(len(points), 1, dtype=torch.int).to(DEVICE),
            "original_size": (image_size, image_size),
        }
        for points in all_points
    ]
    return tensor_images, items


def extract_outputs(outputs: List[Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    pred_logits = []
    pred_cell_probs = []
    for output in outputs:
        point_nums = output["masks"].shape[0]
        for i in range(point_nums):
            pred_logits.append(output["low_res_logits"][i][0])
            pred_cell_probs.append(output["iou_predictions"][i][0])
    return torch.stack(pred_logits).to(DEVICE), torch.stack(pred_cell_probs).to(DEVICE)


def extract_true_masks(
    images: List[np.ndarray],
    cell_masks: List[np.ndarray],
    all_points: List[List[np.ndarray]],
    all_cell_probs: List[List[int]],
    low_res_shape: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    true_masks = []
    true_cell_probs = []
    for image, masks, points, cell_probs in zip(images, cell_masks, all_points, all_cell_probs):
        for mask, point, cell_prob in zip(masks, points, cell_probs):
            low_res_true_mask = cv2.resize(
                mask.astype(np.int32),
                dsize=(low_res_shape[0], low_res_shape[1]),
                interpolation=cv2.INTER_NEAREST_EXACT,
            )
            if low_res_true_mask.max() == 0:
                cell_prob = 0
            true_masks.append(low_res_true_mask)
            true_cell_probs.append(cell_prob)
    true_cell_probs = torch.tensor(true_cell_probs, dtype=torch.float32).to(DEVICE)
    true_masks = torch.tensor(np.array(true_masks), dtype=torch.float32).to(DEVICE)
    return true_masks, true_cell_probs


def is_valid_batch(images: List[np.ndarray], all_points: List[List[np.ndarray]]) -> bool:
    return len(images) > 0 and len(all_points) > 0 and all(len(points) > 0 for points in all_points)


def compute_loss(
    model: LoRA_Sam,
    config: Dict,
    batch_images: List[torch.Tensor],
    batch_points: List[Dict[str, torch.Tensor]],
    cell_masks: List[np.ndarray],
    all_points: List[List[np.ndarray]],
    all_cell_probs: List[List[int]],
) -> torch.Tensor:
    image_embeddings = model.sam.encoder_image_embeddings(batch_images)
    outputs = model.sam.forward_train(
        batched_input=batch_points,
        multimask_output=False,
        input_image_embeddings=image_embeddings,
        image_size=(config["sam_image_size"], config["sam_image_size"]),
    )

    pred_logits, pred_cell_probs = extract_outputs(outputs)
    true_masks, true_cell_prob = extract_true_masks(
        batch_images, cell_masks, all_points, all_cell_probs, pred_logits[0].shape
    )

    ce_loss = cross_entropy_loss(
        true_masks=true_masks,
        pred_logits=pred_logits,
        true_cell_prob=true_cell_prob,
    )
    cell_prob_loss = cell_prob_mse_loss(true_cell_prob=true_cell_prob, pred_cell_prob=pred_cell_probs)
    return cell_prob_loss + ce_loss * config["ce_loss_weight"]


def train_epoch(
    model: LoRA_Sam,
    config: Dict,
    trainloader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: OneCycleLR,
    stop_event=None,
):
    model.train()
    actual_ga_step = 0
    for i_batch, batch_data in enumerate(tqdm(trainloader, desc="Batches", leave=False)):
        if stop_event is not None and stop_event.is_set():
            return
        images, true_instance_masks, cell_masks, all_points, all_cell_probs = batch_data

        if not is_valid_batch(images, all_points):
            continue

        batch_images, batch_points = to_tensor(images, all_points, config["sam_image_size"])

        loss = compute_loss(model, config, batch_images, batch_points, cell_masks, all_points, all_cell_probs)

        actual_ga_step += 1
        loss_ga = loss / (actual_ga_step if (i_batch + 1) == len(trainloader) else config["gradient_accumulation_step"])
        loss_ga.backward()

        if ((i_batch + 1) % config["gradient_accumulation_step"] == 0) or ((i_batch + 1) == len(trainloader)):
            optimizer.step()
            optimizer.zero_grad()
            actual_ga_step = 0
            scheduler.step()


def save_model_pth(model: LoRA_Sam, save_path: str):
    model.save_lora_parameters(save_path)


def main(config_path: Union[str, Dict, Path], save_model: bool = True) -> LoRA_Sam:
    if isinstance(config_path, dict):
        config = config_path
    elif isinstance(config_path, str) or isinstance(config_path, Path):
        with open(config_path) as f:
            config = yaml.safe_load(f)
    set_env(
        config["deterministic"],
        config["seed"],
        config["allow_tf32_on_cudnn"],
        config["allow_tf32_on_matmul"],
    )
    prepare_directories(config)

    train_dataset = load_dataset(config)
    model = load_model(config)
    trainloader, optimizer, scheduler = setup_training(config, model, train_dataset)

    if config["track_gpu_memory"]:
        gpu_memory_tracker = GPUMemoryTracker()
        gpu_memory_tracker.reset()
        memory_stats = {}
    for epoch in tqdm(range(config["epoch_max"]), desc="Epochs"):
        train_epoch(model, config, trainloader, optimizer, scheduler)
        if config["track_gpu_memory"]:
            memory_stats[epoch] = gpu_memory_tracker.get_memory_stats()

    if save_model:
        save_model_pth(model, config["result_pth_path"])

    if config["track_gpu_memory"]:
        with open(Path(config["result_pth_path"]).parent / "memory_stats.json", "w") as f:
            json.dump(memory_stats, f, indent=4)
    return model
