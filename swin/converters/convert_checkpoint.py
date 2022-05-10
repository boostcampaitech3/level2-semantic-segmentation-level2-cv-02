# Modified from
# https://github.com/open-mmlab/mmsegmentation/blob/master/tools/model_converters/swin2mmseg.py

import os.path as osp
from collections import OrderedDict

import mmcv
import torch
from mmcv.runner import CheckpointLoader


def convert_swin(ckpt):
    new_ckpt = OrderedDict()

    def correct_unfold_reduction_order(x):
        out_channel, in_channel = x.shape
        x = x.reshape(out_channel, 4, in_channel // 4)
        x = x[:, [0, 2, 1, 3], :].transpose(1, 2).reshape(out_channel, in_channel)
        return x

    def correct_unfold_norm_order(x):
        in_channel = x.shape[0]
        x = x.reshape(4, in_channel // 4)
        x = x[[0, 2, 1, 3], :].transpose(0, 1).reshape(in_channel)
        return x

    for k, v in ckpt.items():
        if 'downsample' in k:
            if 'reduction.' in k:
                new_v = correct_unfold_reduction_order(v)
            elif 'norm.' in k:
                new_v = correct_unfold_norm_order(v)

        new_ckpt['backbone.' + k] = v

    return new_ckpt


if __name__ == '__main__':
    src = '/opt/ml/input/code/configs/swin_large_patch4_window7_224_22k.pth'
    dst = '/opt/ml/input/code/configs/revised.pth'

    checkpoint = CheckpointLoader.load_checkpoint(src, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    weight = convert_swin(state_dict)
    mmcv.mkdir_or_exist(osp.dirname(dst))
    torch.save(weight, dst)
