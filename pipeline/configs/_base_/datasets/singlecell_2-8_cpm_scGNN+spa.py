# dataset settings
dataset_type = 'SCDataset'
data_root = '/scratch/scdata/data/sc/'
img_norm_cfg = dict(
    mean=[204.973, 198.492, 200.727], std=[69.245, 77.032, 73.532],
    to_rgb=True)
crop_size = (400, 400)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(600, 600), ratio_range=(1.0, 1.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomRotate', prob=0.3, degree=180),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        img_scale=(600, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            # dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]
    )
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_root=data_root,

        img_dir=['151507', '151509', '151510',
                 '151669', '151670', '151671', '151672',
                 '151673', '151674', '151675', '2-8',
                 '151676', 'T4857', '151508', '18-64'
                 ],
        ann_dir=['151507_label', '151509_label', '151510_label',
                 '151669_label', '151670_label', '151671_label', '151672_label',
                 '151673_label', '151674_label', '151675_label', '2-8_label',
                 '151676_label', 'T4857_label', '151508_label', '18-64_label'
                 ],

        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,

        img_dir='2-5',
        ann_dir='2-5_label',

        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,

        img_dir='2-5',
        ann_dir='2-5_label',

        pipeline=test_pipeline))
