# Modified from
# https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation/blob/main/configs/_base_/datasets/ade20k.py

dataset_type = 'UpstageDataset'
data_root = '/opt/ml/input/data'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=(512, 512)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(256, 256),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        img_dir=data_root + '/mmseg/images/train',
        ann_dir=data_root + '/mmseg/annotations/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        img_dir=data_root + '/mmseg/images/validation',
        ann_dir=data_root + '/mmseg/annotations/validation',
        pipeline=test_pipeline,
        test_mode=True),
    test=dict(
        type=dataset_type,
        img_dir=data_root + '/mmseg/test',
        pipeline=test_pipeline,
        test_mode=True))
