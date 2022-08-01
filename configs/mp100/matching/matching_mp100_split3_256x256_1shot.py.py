log_level = 'INFO'
load_from = None
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
checkpoint_config = dict(interval=10)
evaluation = dict(interval=10, metric='PCK', key_indicator='PCK', gpu_collect=True)
optimizer = dict(
    type='Adam',
    lr=5e-4,
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[170, 200])
total_epochs = 210
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

channel_cfg = dict(
    num_output_channels=1,
    dataset_joints=1,
    dataset_channel=[
        [0, ],
    ],
    inference_channel=[
        0,
    ])

# model settings
model = dict(
    type='FewShotKeypoint',
    pretrained='torchvision://resnet50',
    encoder_sample=dict(type='ResNet', depth=50),
    encoder_query=dict(type='ResNet', depth=50),
    keypoint_head=dict(
        type='MatchingHead',
        in_channels=2048,
        out_channels=1,
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=False,
        post_process='default',
        shift_heatmap=True,
        modulate_kernel=11))

data_cfg = dict(
    image_size=[256, 256],
    heatmap_size=[64, 64],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel']
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=15, scale_factor=0.15),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTarget', sigma=2),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs', 'category_id'
        ]),
]

valid_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownAffineFewShot'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTargetFewShot', sigma=2),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'center', 'scale', 'rotation', 'bbox_score',
            'flip_pairs', 'category_id'
        ]),
]

test_pipeline = valid_pipeline

data_root = 'data/mp100'
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type='FewShotKeypointDataset',
        ann_file=f'{data_root}/annotations/mp100_split3_train.json',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        valid_class_ids=None,
        num_shots = 1,
        pipeline=train_pipeline),
    val=dict(
        type='FewShotKeypointDataset',
        ann_file=f'{data_root}/annotations/mp100_split3_val.json',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        valid_class_ids=None,
        num_shots = 1,
        num_queries = 15,
        num_episodes = 100,
        pipeline=valid_pipeline),
    test=dict(
        type='FewShotKeypointDataset',
        ann_file=f'{data_root}/annotations/mp100_split3_test.json',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        valid_class_ids=None,
        num_shots = 1,
        num_queries = 15,
        num_episodes = 200,
        pipeline=valid_pipeline),
)

shuffle_cfg = dict(interval=1)
