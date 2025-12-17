# OWODB settings
owod_settings = {
    # 4 tasks
    "MOWODB": {
        "task_list": [0, 20, 40, 60, 80],
        "test_image_set": "all_task_test"
    },
    # 4 tasks
    "SOWODB": {
        "task_list": [0, 19, 40, 60, 80],
        "test_image_set": "test",
    },
    # 3 tasks
    "nuOWODB": {
        "task_list": [0, 10, 17, 23],
        "test_image_set": "test",
    },
    # "ODinW13" : {
    #     "task_list" : [0,5,12,13,14,16,17,35,36,39,42,43,44,44],
    #     "test_image_set": "test",
    # },
    "ODinW13" : {
        "task_list" : [0,5,7,1,1,2,1,20,1,3,5,1,1,2],
        "test_image_set": "test",
    },
    "ZCOCO" :{
        "task_list" : [0,80],
        "test_image_set" : "test_all"
    }
}

owod_root = "/root/data-tmp/odinw13"

# Configs from environment variables
owod_dataset = '{{$DATASET:ODinW13}}'                                             # dataset name (default: MOWODB)
owod_task = {{'$TASK:1'}}                                                         # task number (default: 1)
train_image_set = '{{$IMAGESET:train}}'                                           # owod train image set (default: train)

threshold = {{'$THRESHOLD:0.05'}}                                                 # prediction score threshold for known class (default: 0.05)
training_strategy = {{'$TRAINING_STRATEGY:0'}}                                    # 0: OWOD, 1: ORACLE (default: 0)
save_rets = {{'$SAVE:False'}}                                                     # save evaluation results to 'eval_output.txt' (default: False)

class_text_path = f"{owod_root}/ImageSets/{owod_dataset}/t{owod_task}train.txt"  # text inputs path for open-vocabulary model 
test_image_set = owod_settings[owod_dataset]['test_image_set']                    # owod test image set

task_list = owod_settings[owod_dataset]['task_list']
PREV_INTRODUCED_CLS = task_list[owod_task - 1]                                    # previous known classes number
CUR_INTRODUCED_CLS = task_list[owod_task] - task_list[owod_task - 1]              # current known classes number
META_PATH = f'/root/data-tmp/odinw13/{owod_dataset}_task_metadata.json'

# OWOD config
owod_cfg = dict(
    split=test_image_set,
    task_num=owod_task,
    PREV_INTRODUCED_CLS=PREV_INTRODUCED_CLS,
    CUR_INTRODUCED_CLS=CUR_INTRODUCED_CLS,
    num_classes=PREV_INTRODUCED_CLS + CUR_INTRODUCED_CLS + 1,
)

# OWOD dataset
owod_train_dataset = dict(
    type='MultiModalOWDataset',
    dataset=dict(
        type='OWODDataset',
        data_root=owod_root,
        image_set=train_image_set,
        dataset=owod_dataset,
        task_id=owod_task,
        task_metadata_path=META_PATH,
        training_strategy=training_strategy,
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path=class_text_path,)

owod_val_dataset = dict(
    type='MultiModalOWDataset',
    dataset=dict(type='OWODDataset',
                 data_root=owod_root,
                 image_set=test_image_set,
                 dataset=owod_dataset,
                 task_id=owod_task,
                 task_metadata_path=META_PATH,
                 test_mode=True),
    class_text_path=class_text_path,)

# OWOD evaluator
owod_val_evaluator = dict(
    type='OpenWorldMetric',
    data_root=owod_root,
    dataset_name=owod_dataset,
    task_id=owod_task,
    split=owod_cfg["split"],
    task_metadata_path=META_PATH,
    threshold=threshold,
    save_rets=save_rets,
    owod_cfg=owod_cfg,
)

# owod_test_dataset = dict(
#     type='MultiModalOWDataset',
#     dataset=dict(type='OWODDataset',
#                  data_root=owod_root,
#                  image_set=test_image_set,
#                  dataset='ZCOCO',
#                  task_id=owod_task,
#                  task_metadata_path=META_PATH,
#                  test_mode=True),
#     class_text_path=class_text_path,)
