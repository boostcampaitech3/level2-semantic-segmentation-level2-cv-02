# Modified from
# https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation/blob/main/configs/_base_/schedules/schedule_160k.py

optimizer_config = dict()
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1e-3,
    step=[40, 50])
runner = dict(type='EpochBasedRunner', max_epochs=60)
evaluation = dict(interval=1, metric='mIoU')
