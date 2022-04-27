# Modified from
# https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation/blob/main/configs/swin/upernet_swin_base_patch4_window7_512x512_160k_ade20k.py

_base_ = [
    '/opt/ml/input/code/configs/upernet_swin.py',
    '/opt/ml/input/code/configs/coco_simillar_dataset.py',
    '/opt/ml/input/code/configs/schedule_160k.py',
    '/opt/ml/input/code/configs/default_runtime.py'
]

work_dir = '/opt/ml/input/code'

model = dict(
    backbone=dict(
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        ape=False,
        drop_path_rate=0.3,
        patch_norm=True,
        use_checkpoint=False
    ),
    decode_head=dict(
        in_channels=[128, 256, 512, 1024],
        num_classes=11
    ),
    auxiliary_head=dict(
        in_channels=512,
        num_classes=11
    ))

optimizer = dict(type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))

seed = 2022
gpu_ids = [0]
