# --------------------------------------------------------
# PODA: Prompt-driven Zero-shot Domain Adaptation
# Copyright (c) 2024 valeo.ai, astra-vision 
#
# Written by Tuan-Hung Vu
# --------------------------------------------------------

_base_ = [
    './faster_rcnn_r50_fpn_pretrainedCLIP.py',
    './cityscapes_detection.py',
    '../_base_/default_runtime.py'
]
model = dict(
    backbone=dict(type='ModifiedResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=4,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        target_domain = 'fog',
        augmented_layer = 1,
        mixing_style = False,
        init_cfg=None),
    roi_head=dict(
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=8,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))))
# optimizer
# lr is set for a batch size of 8
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[7])
runner = dict(
    type='EpochBasedRunner', max_epochs=8) 
log_config = dict(interval=100)
load_from = './work_dirs/faster_rcnn_r50_fpn_1x_pretrainedCLIP_cityscapes/latest.pth'  # noqa

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (1 samples per GPU)
auto_scale_lr = dict(base_batch_size=8)

## override for testing
dataset_type = 'CityscapesDataset'
data_root = 'data/cityscapes/'
data_root_target = 'data/cityscapes_foggy/val/'
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    val=dict(
        type=dataset_type,
        ann_file=data_root +
        'annotations/instancesonly_filtered_gtFine_val.json',
        img_prefix=data_root_target),
    test=dict(
        type=dataset_type,
        ann_file=data_root +
        'annotations/instancesonly_filtered_gtFine_val.json',
        img_prefix=data_root_target))

evaluation=dict(classwise=True, iou_thrs=[0.5], metric='bbox')