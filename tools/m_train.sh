export PYTHONPATH=$PYTHONPATH:.
export HF_HOME="/root/data-tmp/.cache"
export CUDA_VISIBLE_DEVICES=0

python tools/train_owod_tasks.py \
    ODinW13 \
    ./configs/pretrain/yolo_world_UNI_v2_l_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py \
    ./pretrained/l_stage1-7d280586.pth