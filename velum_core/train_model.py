import json
from datetime import datetime
from pathlib import Path


def _save_config_sidecar(config, loss_history):
    """Save a reproducible JSON sidecar next to the checkpoint."""
    pth = Path(config["result_pth_path"])
    sidecar = pth.with_suffix(".json")
    serialisable = {
        k: v for k, v in config.items()
        if isinstance(v, (int, float, str, bool, list))
    }
    serialisable["loss_history"] = loss_history
    serialisable["saved_at"] = datetime.now().isoformat()
    with open(sidecar, "w") as f:
        json.dump(serialisable, f, indent=2)


def load_dataset(config):
    from data.dataset import TrainDataset
    return TrainDataset(
        image_dir=config["train_image_dir"],
        mask_dir=config["train_mask_dir"],
        resize_size=config["resize_size"],
        patch_size=config["patch_size"],
        train_id=config["train_id"],
        duplicate_data=config["duplicate_data"],
    )


def train_model(config, state_manager, progress_queue=None, stop_event=None):
    """Train the model.

    progress_queue: if provided, puts dicts {"epoch", "pct", "loss"} for in-process UI updates.
                    Also used for file-based Streamlit state (both can coexist).
    stop_event: threading.Event — set it to request early stop without touching disk.
    """
    import os

    selected_device = config.get("selected_device", "cpu")
    if selected_device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif selected_device == "mps":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = selected_device

    from cellseg1_train import (
        load_model,
        prepare_directories,
        save_model_pth,
        setup_training,
        train_epoch,
    )
    from set_environment import set_env

    set_env(
        config["deterministic"],
        config["seed"],
        config["allow_tf32_on_cudnn"],
        config["allow_tf32_on_matmul"],
    )
    prepare_directories(config)

    train_dataset = load_dataset(config)
    model = load_model(config)
    # Resume from existing LoRA checkpoint (used by refine flow)
    finetune_from = config.get("finetune_from")
    if finetune_from and Path(finetune_from).exists():
        model.load_lora_parameters(finetune_from)
    trainloader, optimizer, scheduler = setup_training(config, model, train_dataset)

    state_manager.clear_loss_history()
    loss_history = []
    save_model = config["result_pth_path"]
    started_at = datetime.now().isoformat()

    from velum_core import experiment_tracking as tracking
    tracked = tracking.start_run("train", config)

    try:
        for epoch in range(config["epoch_max"]):
            stopped = (
                (stop_event is not None and stop_event.is_set())
                or state_manager.check_stop_flag()
            )
            if stopped:
                save_model = False
                break

            avg_loss = train_epoch(model, config, trainloader, optimizer, scheduler)
            current_epoch = epoch + 1
            pct = int(current_epoch / config["epoch_max"] * 100)
            loss_entry = {"epoch": current_epoch, "loss": round(float(avg_loss), 6)}
            loss_history.append(loss_entry)
            tracked.track(loss_entry["loss"], name="loss", step=current_epoch)

            # fast in-process path
            if progress_queue is not None:
                progress_queue.put({"epoch": current_epoch, "pct": pct, "loss": loss_entry["loss"]})

            # file-based path (Streamlit GUI compatibility)
            state_manager.save_loss_history(loss_history)
            state_manager.save_progress(pct, current_epoch)

        if save_model:
            save_model_pth(model, config["result_pth_path"])
            _save_config_sidecar(config, loss_history)

        final_loss = loss_history[-1]["loss"] if loss_history else None
        tracked["status"] = "completed" if save_model else "stopped"
        state_manager.append_history_entry({
            "started_at": started_at,
            "finished_at": datetime.now().isoformat(),
            "sam_type": config.get("vit_name", ""),
            "lora_rank": config.get("image_encoder_lora_rank", ""),
            "epochs_run": len(loss_history),
            "epoch_max": config.get("epoch_max", ""),
            "final_loss": final_loss,
            "checkpoint": config.get("result_pth_path", ""),
            "status": "completed" if save_model else "stopped",
        })
    finally:
        tracked.close()
        state_manager.clear_training_state()
