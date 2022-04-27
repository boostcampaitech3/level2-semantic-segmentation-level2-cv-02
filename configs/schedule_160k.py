# Modified from
# https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation/blob/main/configs/_base_/schedules/schedule_160k.py

optimizer_config = dict()
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
runner = dict(type='EpochBasedRunner', max_epochs=60)
evaluation = dict(interval=1, metric='mIoU')
