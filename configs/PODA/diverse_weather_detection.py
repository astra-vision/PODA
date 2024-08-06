# --------------------------------------------------------
# PODA: Prompt-driven Zero-shot Domain Adaptation
# Copyright (c) 2024 valeo.ai, astra-vision 
#
# Written by Tuan-Hung Vu
# --------------------------------------------------------

# dataset settings
dataset_type = 'DiverseWeatherDataset'
data_root = 'data/diverse_weather/'
img_norm_cfg = dict(
    mean=[122.771, 116.746, 104.094], std=[68.501, 66.632, 70.323], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize', img_scale=[(2048, 800), (2048, 1024)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            min_size=1,
            ann_file=[
                data_root + 'daytime_clear_new/VOC2007/ImageSets/Main/train.txt',
            ],
            img_prefix=[data_root + 'daytime_clear_new/VOC2007/'],
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'daytime_clear_new/VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'daytime_clear_new/VOC2007/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'daytime_clear_new/VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'daytime_clear_new/VOC2007/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='mAP')
