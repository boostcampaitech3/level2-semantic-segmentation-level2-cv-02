#
# boostcamp AI Tech
# Trash Semantic Segmentation Competition
#


import torch

from mmcv.runner import load_checkpoint
from mmseg.models import build_segmentor
from mmseg.apis import train_segmentor, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmcv.parallel import MMDataParallel

import wandb

from options import WANDB_RUN, CONFIG_PATH
from options import seed_all, wandb_init, get_cfg, make_predictions


BASE_EPOCHS = 0
EPOCHS = 100 - BASE_EPOCHS

if __name__ == '__main__':
    # Init
    seed_all(42)
    wandb_init()

    cfg = get_cfg(CONFIG_PATH, EPOCHS)

    model = build_segmentor(cfg.model)
    if BASE_EPOCHS == 0:
        # Learning from the pretrained weights
        checkpoint_path = "/opt/ml/input/code/configs/revised.pth"
        checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
        model = MMDataParallel(model.cuda(), device_ids=[0])
    else:
        # Learning from the checkpoint
        assert BASE_EPOCHS > EPOCHS, "The original checkpoint will be overwritten."
        assert BASE_EPOCHS > cfg.lr_config.step[1], "Learning rate should be considered carefully."
        cfg.optimizer.lr = 1e-6
        cfg.lr_config = None

        checkpoint_path = f"./epoch_{BASE_EPOCHS}.pth"
        checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
        model = MMDataParallel(model.cuda(), device_ids=[0])

    datasets = [build_dataset(cfg.data.train), build_dataset(cfg.data.test)]

    # Train
    wandb.alert(title="Train Started", text=f"{WANDB_RUN}")
    train_segmentor(model, datasets[0], cfg, distributed=False, validate=True)

    # Prediction
    checkpoint_path = f"./epoch_{EPOCHS}.pth"

    model = build_segmentor(cfg.model)
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
    model = MMDataParallel(model.cuda(), device_ids=[0])

    data_loader = build_dataloader(
        datasets[1],
        samples_per_gpu=1,
        workers_per_gpu=8,
        dist=False,
        shuffle=False
    )

    output = single_gpu_test(model, data_loader)
    torch.save(output, './output.tmp')
    make_predictions(output, cfg, f"./epoch{EPOCHS + BASE_EPOCHS}.csv")
