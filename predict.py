import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from data.utils import (
    keep_largest_connected_component,
    read_image_to_numpy,
    remap_mask_color,
    resize_image,
)
from peft.sam_lora_image_encoder_mask_decoder import LoRA_Sam
from segment_anything import SamAutomaticMaskGeneratorOptMaskNMS, sam_model_registry
from set_environment import DEVICE, set_env


def sam_output_to_mask(output):
    mask = np.zeros_like(output[0]["segmentation"], dtype=np.int64)
    output = sorted(output, key=lambda x: -x["area"])
    for i, o in enumerate(output):
        mask[o["segmentation"]] = 0
        mask += (i + 1) * keep_largest_connected_component(o["segmentation"])
    mask = remap_mask_color(mask)
    return mask


def predict_images(config, images, progress_callback=None, stop_event=None):
    set_env(
        config["deterministic"],
        config["seed"],
        config["allow_tf32_on_cudnn"],
        config["allow_tf32_on_matmul"],
    )
    model = load_model_from_config(config, empty_lora=False)
    model.eval()
    if hasattr(model, "sam"):
        model_sam = model.sam
    else:
        model_sam = model

    mask_generator = SamAutomaticMaskGeneratorOptMaskNMS(
        model=model_sam,
        points_per_side=config["points_per_side"],
        points_per_batch=config["points_per_batch"],
        crop_n_layers=config["crop_n_layers"],
        crop_n_points_downscale_factor=config["crop_n_points_downscale_factor"],
        box_nms_thresh=config["box_nms_thresh"],
        crop_nms_thresh=config["crop_nms_thresh"],
        pred_iou_thresh=config["pred_iou_thresh"],
        min_mask_region_area=config["min_mask_region_area"],
        max_mask_region_area_ratio=config["max_mask_region_area_ratio"],
        stability_score_thresh=config["stability_score_thresh"],
        stability_score_offset=config["stability_score_offset"],
    )

    pred_masks = []
    with torch.no_grad():
        for i, image in enumerate(tqdm(images, disable=progress_callback is not None)):
            if stop_event and stop_event.is_set():
                break
            output = mask_generator.generate(image)
            if output == []:
                mask = np.zeros_like(image[:, :, 0], dtype=np.uint16)
            else:
                mask = sam_output_to_mask(output)
            pred_masks.append(mask)

            if progress_callback:
                progress = i
                progress_callback(progress)

    return pred_masks


def load_model_from_config(config, empty_lora=False):
    model = sam_model_registry[config["vit_name"]](
        checkpoint=config["model_path"], image_size=config["sam_image_size"]
    )
    model = LoRA_Sam(model, config)
    model = model.to(DEVICE)
    if empty_lora:
        pass
    else:
        model.load_lora_parameters(Path(config["result_pth_path"]))
    return model


def predict_config(config, test_image_folder=None, result_folder=None, save=True):
    set_env(
        config["deterministic"],
        config["seed"],
        config["allow_tf32_on_cudnn"],
        config["allow_tf32_on_matmul"],
    )
    if test_image_folder is None:
        image_path = Path(config["data_dir"]) / "test/images"
    else:
        image_path = Path(test_image_folder)

    image_files = sorted(list(Path(image_path).iterdir()))
    image_file_names = [i.stem for i in image_files]

    images = [read_image_to_numpy(i) for i in image_files]
    images = [resize_image(i, config["resize_size"]) for i in images]

    pred_masks = predict_images(config, images)

    if save:
        if result_folder is None:
            save_folder = Path(config["result_pth_path"]).parent / "pred_masks"
        else:
            save_folder = Path(result_folder)
        if save_folder.exists() and save_folder.is_dir():
            try:
                shutil.rmtree(save_folder)
                print(f"Existing folders have been deleted: {save_folder}")
            except Exception as e:
                print(f"Unable to delete the folder {save_folder}: {e}")

        save_folder.mkdir(exist_ok=True, parents=True)

        for i, mask in enumerate(pred_masks):
            if mask.dtype != np.uint16:
                mask = mask.astype(np.uint16)

            save_path = save_folder / f"{image_file_names[i]}.png"
            success = cv2.imwrite(str(save_path), mask)

            if not success:
                print(f"Failed to save the file: {save_path}")

    return pred_masks
