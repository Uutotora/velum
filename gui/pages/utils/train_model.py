from data.dataset import TrainDataset


def load_dataset(config):
    train_dataset = TrainDataset(
        image_dir=config["train_image_dir"],
        mask_dir=config["train_mask_dir"],
        resize_size=config["resize_size"],
        patch_size=config["patch_size"],
        train_id=config["train_id"],
        duplicate_data=config["duplicate_data"],
    )
    return train_dataset


def train_model(config, state_manager):
    import os

    selected_device = config.get("selected_device", "cpu")
    if selected_device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        print("INFO: Running training on CPU.")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = selected_device
        print(f"INFO: Running training on GPU {selected_device}.")

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

    trainloader, optimizer, scheduler = setup_training(config, model, train_dataset)

    save_model = config["result_pth_path"]
    try:
        for epoch in range(config["epoch_max"]):
            if state_manager.check_stop_flag():
                save_model = False
                break

            train_epoch(model, config, trainloader, optimizer, scheduler)
            progress = int(((epoch + 1) / config["epoch_max"]) * 100)
            current_epoch = epoch + 1

            state_manager.save_progress(progress, current_epoch)
        if save_model:
            save_model_pth(model, config["result_pth_path"])
    finally:
        state_manager.clear_training_state()
