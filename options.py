import torch

from mmcv import Config

from pycocotools.coco import COCO
import numpy as np
import pandas as pd

import os
import random
import mmcv
from mmseg.apis.test import np2tmp
from skimage.measure import block_reduce



def make_predictions(output, idx_order, cfg, loc: str) -> None:
    prediction_strings = []
    file_names = []

    coco = COCO(os.path.join(cfg.data_root, 'test.json'))

    for i in range(len(output)):
        image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
        current_idx = image_info['file_name'].replace('/', '_')
        output_idx = idx_order.index(current_idx)
        out = output[output_idx]
        prediction_string = ' '.join(str(pred) for pred in out)
        prediction_string = prediction_string.replace('[', '').replace(']', '')
        file_names.append(image_info['file_name'])
        prediction_strings.append(prediction_string)

    submission = pd.DataFrame()
    submission['image_id'] = file_names
    submission['PredictionString'] = prediction_strings
    submission.to_csv(loc, index=False)
    
    
def single_gpu_test_only(model, data_loader):
    model.eval()
    results = []
    idx_order = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    loader_indices = data_loader.batch_sampler

    for batch_indices, data in zip(loader_indices, data_loader):
        with torch.no_grad():
            # result = model(return_loss=False, **data)
            result_before = model(return_loss=False, **data)
            result = block_reduce(np.array(result_before), (1,2,2), np.max)
            current_idx = data['img_metas'][0].data[0][0]['ori_filename']
            
        results.extend(result)
        idx_order.append(current_idx)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()

    return results, idx_order