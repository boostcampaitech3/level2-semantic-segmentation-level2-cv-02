# Modified from
# https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation/blob/main/configs/_base_/default_runtime.py

checkpoint_config = dict(max_keep_ckpts=1000, interval=4)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook',
            interval=100,
            init_kwargs=dict()
        )
    ])

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
