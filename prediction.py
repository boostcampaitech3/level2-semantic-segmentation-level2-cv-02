#
# boostcamp AI Tech
# Trash Semantic Segmentation Competition
#


from mmcv.runner import load_checkpoint
from mmseg.models import build_segmentor
from mmseg.apis import single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmcv.parallel import MMDataParallel

import argparse

from options import get_cfg, make_predictions


if __name__ == '__main__':
    # Init
    parser = argparse.ArgumentParser()
    parser.add_argument('epoch', type=int)
    args = parser.parse_args()

    cfg = get_cfg(args.epoch)

    dataset = build_dataset(cfg.data.test)

    # Prediction
    checkpoint_path = f"./epoch_{args.epoch}.pth"

    model = build_segmentor(cfg.model)
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
    model = MMDataParallel(model.cuda(), device_ids=[0])

    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=8,
        dist=False,
        shuffle=False
    )

    output = single_gpu_test(model, data_loader)
    make_predictions(output, cfg, f"./epoch{args.epoch}.csv")
