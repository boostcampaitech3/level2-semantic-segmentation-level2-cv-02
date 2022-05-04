#
# boostcamp AI Tech
# Trash Semantic Segmentation Competition
#


import torch

from mmcv import Config

from pycocotools.coco import COCO
import numpy as np
import pandas as pd

import os
import random

import wandb


WANDB_PROJECT = "trash_segmentation_nestiank"
WANDB_ENTITY = "bucket_interior"
WANDB_RUN = "Swin_Checkpoint"

CONFIG_PATH = '/opt/ml/input/code/configs/modified_swin_large.py'


def seed_all(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def wandb_init() -> None:
    wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, name=WANDB_RUN)


def get_cfg(epochs: int):
    cfg = Config.fromfile(CONFIG_PATH)
    cfg.log_config.hooks[1].init_kwargs.project = WANDB_PROJECT
    cfg.log_config.hooks[1].init_kwargs.entity = WANDB_ENTITY
    cfg.log_config.hooks[1].init_kwargs.name = WANDB_RUN
    cfg.lr_config.step = [int(epochs * 0.7), int(epochs * 0.8)]
    cfg.runner.max_epochs = epochs
    return cfg


def make_predictions(output, cfg, loc: str) -> None:
    prediction_strings = []
    file_names = []

    coco = COCO(os.path.join(cfg.data_root, 'test.json'))

    for i, out in enumerate(output):
        if i % 50 == 0:
            print('Iteration', i + 1)

        image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
        prediction_string = ' '.join(str(pred) for pred in out)
        prediction_string = prediction_string.replace('[', '').replace(']', '')
        prediction_string = prediction_string.replace('"', '').replace('\n', '')

        file_names.append(image_info['file_name'])
        prediction_strings.append(prediction_string)

    submission = pd.DataFrame()
    submission['image_id'] = file_names
    submission['PredictionString'] = prediction_strings
    submission.to_csv(loc, index=False)
