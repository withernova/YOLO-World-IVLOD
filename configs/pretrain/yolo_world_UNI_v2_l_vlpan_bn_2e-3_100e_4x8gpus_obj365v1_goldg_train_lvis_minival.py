_base_ = [
    '../../third_party/mmyolo/configs/yolov8/yolov8_l_syncbn_fast_8xb16-500e_coco.py', # 建议基座切换回 v8 以匹配 YOLO-World
    '../datasets/odinw_dataset.py'
]

custom_imports = dict(imports=['yolo_world'], allow_failed_imports=True)

EXP_NAME = "第十七次测试-尝试迁移YOLO-World"
WORK_DIR = "/root/data-fs/YOLOWorld"

num_classes = _base_.PREV_INTRODUCED_CLS + _base_.CUR_INTRODUCED_CLS 
num_training_classes = _base_.PREV_INTRODUCED_CLS + _base_.CUR_INTRODUCED_CLS
max_epochs = 120
close_mosaic_epochs = 2
save_epoch_intervals = 5
val_interval = 20
val_interval_stage2 = 20

text_channels = 512
neck_embed_channels = [128, 256, _base_.last_stage_out_channels // 2]
neck_num_heads = [4, 8, _base_.last_stage_out_channels // 2 // 32]

base_lr = 2e-3 
weight_decay = 0.05 / 2
train_batch_size_per_gpu = 16

ISCOOP = True
CKPT_PATH = f'embeddings/uniow-w/{_base_.owod_dataset}_class_Co_embeddings.pth' if ISCOOP else f'embeddings/uniow-w/{_base_.owod_dataset}_class_embeddings.pth'
CKPT_RUNNING = 'embeddings/uniow-w/running_class_embeddings.pth'
CKPT_FINAL = f'{WORK_DIR}/{EXP_NAME}/final.pth'

load_from = 'pretrained/l_stage1-7d280586.pth' 

# trainable (1), frozen (0)
embedding_mask = ([0] * _base_.PREV_INTRODUCED_CLS +    # previous classes
                  [1] * _base_.CUR_INTRODUCED_CLS  +    # current class
                  [1]                              +    # unknown class
                  [1])                                  # anchor class

# --- Model Settings ---
model = dict(
    type='OWODDetector', # 保持你的 Detector 类，但内部需要适配 mm_neck=True
    mm_neck=True,        # 开启 Neck，这是迁移到 YOLO-World 的关键
    prompt_dim=512,
    freeze_prompt=False,
    task_id=_base_.owod_task,
    task_metadata_path=_base_.META_PATH,
    all_class_embeddings_path=CKPT_RUNNING,
    mode = CKPT_FINAL if _base_.owod_dataset == 'ZCOCO' else 'train',
    
    # 使用 YOLO-World 的预处理器
    data_preprocessor=dict(type='YOLOv5DetDataPreprocessor'),
    
    backbone=dict(
        _delete_=True,
        type='MultiModalYOLOBackbone',
        image_model={{_base_.model.backbone}},
        text_model=dict(
            type='HuggingCLIPLanguageBackbone',
            model_name='openai/clip-vit-base-patch32',
        ),
        with_text_model=True,
        frozen_stages=4,
    ),
    
    neck=dict(
        type='YOLOWorldPAFPN',
        guide_channels=text_channels,
        embed_channels=neck_embed_channels,
        num_heads=neck_num_heads,
        block_cfg=dict(type='MaxSigmoidCSPLayerWithTwoConv'),
        freeze_all=True,
    ),
    
    bbox_head=dict(
        type='YOLOWorldHead',
        head_module=dict(
            type='YOLOWorldHeadModule',
            use_bn_head=True,
            embed_dims=text_channels,
            num_classes=num_training_classes,
            freeze_all=True,
        ),
    ),
    
    train_cfg=dict(
        assigner=dict(
            type='BatchTaskAlignedAssigner',
            num_classes=num_training_classes,
            topk=10,
            alpha=0.5,
            beta=6.0,
            eps=1e-9
        )
    ),
    test_cfg=dict(
        unknown_nms=dict(iou_threshold=0.99, score_threshold=0.2), 
        max_per_img=300,
        multi_label=True,
        nms_pre=30000,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.7),
    ),
)

owod_train_dataset = dict(
    _delete_=True,
    **_base_.owod_train_dataset,
    pipeline=_base_.train_pipeline
)

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    collate_fn=dict(type='yolow_collate'), 
    dataset=owod_train_dataset
)

owod_val_dataset = dict(
    _delete_=True,
    **_base_.owod_val_dataset,
    pipeline=_base_.test_pipeline
)

val_dataloader = dict(dataset=owod_val_dataset)
test_dataloader = val_dataloader

val_evaluator = dict(
    _delete_=True,
    **_base_.owod_val_evaluator,
)
test_evaluator = val_evaluator

# --- Training Settings ---
default_hooks = dict(
    param_scheduler=dict(max_epochs=max_epochs),
    checkpoint=dict(
        interval=save_epoch_intervals,
        save_best=['standard_eval/mAP'],
        rule='greater'
    )
)

custom_hooks = [
    dict(type='EMAHook',
         ema_type='ExpMomentumEMA',
         momentum=0.0001,
         update_buffers=True,
         strict_load=False,
         priority=49),
    dict(type='mmdet.PipelineSwitchHook',
         switch_epoch=max_epochs - close_mosaic_epochs,
         switch_pipeline=_base_.train_pipeline_stage2),
    # dict(
    #     type='TensorBoardLossHook',
    #     task_id=EXP_NAME,
    #     log_interval=10
    # ),
    # dict(
    #     type='OWODMetricHook',
    #     json_path=f'{WORK_DIR}/{EXP_NAME}/metrics_summary.json',
    #     task_id=_base_.owod_task,
    #     metric_key='standard_eval/mAP', 
    #     save_best=True
    # )
]

train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=val_interval,
    dynamic_intervals=[((max_epochs - close_mosaic_epochs), val_interval_stage2)]
)

# 优化器配置：YOLO-World 对 Text Model 和 Logit Scale 有特殊处理
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=base_lr,
        weight_decay=weight_decay,
        batch_size_per_gpu=train_batch_size_per_gpu
    ),
    paramwise_cfg=dict(
        bias_decay_mult=0.0,
        norm_decay_mult=0.0,
        custom_keys={
            'backbone.text_model': dict(lr_mult=0.01), # 文本塔通常学习率要低
            'logit_scale': dict(weight_decay=0.0),
            'embeddings': dict(weight_decay=weight_decay)
        }
    ),
    constructor='YOLOWv5OptimizerConstructor'
)