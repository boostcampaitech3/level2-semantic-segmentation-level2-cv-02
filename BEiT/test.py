# mmsegmentation/test.py

from mmcv.runner import load_checkpoint
from mmseg.models import build_segmentor
from mmseg.apis import single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmcv.utils import Config
from mmcv.parallel import MMDataParallel
from mmseg.utils import get_root_logger
from mmseg.models.utils.checkpoint_process import put_checkpoint

import argparse

from options import make_predictions, single_gpu_test_only


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--config',
                        default='/opt/ml/input/code/mmsegmentation/configs/beit/upernet_beit-large_modified.py',
                        help='test config file path')
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.test)
    dataset.test_mode=True

    model = build_segmentor(
        cfg.model,
        test_cfg=cfg.get('test_cfg')
    )
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = MMDataParallel(model.cuda(), device_ids=[0])

    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False
    )

    output, idx_order = single_gpu_test_only(model, data_loader)
    make_predictions(output, idx_order, cfg, f"/opt/ml/input/code/result.csv")