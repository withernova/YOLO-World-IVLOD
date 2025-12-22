from pathlib import Path
import glob
import numpy as np
import torch

from mmengine.config import Config
from mmengine.runner import Runner


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Extract text features')
    parser.add_argument('--config', type=str)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--wildcard', type=str, default='object', help='Wildcard to extract features from')
    parser.add_argument('--save_path', type=str, default='embeddings', help='Save path for extracted features')
    parser.add_argument('--extract_tuned', action='store_true', help='Extract tuned wildcard embeddings')
    return parser.parse_args()


@torch.inference_mode()
def extract_feats(model, dataset=None, task=None, save_path='embeddings'):
    prompt = []
    prompt_path = f'data/odinw/ImageSets/{dataset}/t{task}_known.txt'
    with open(prompt_path, 'r') as f:
        prompt = [line.strip() for line in f.readlines()]
    save_path = save_path / f'{dataset.lower()}_t{task}.npy'

    text_feats = model.backbone.forward_text([prompt]).squeeze(0).detach().cpu()
    print(f"Extracted text features from {dataset}/Task({task}):", text_feats.shape)

    np.save(save_path, text_feats.numpy())

import json
@torch.inference_mode()
def extract_feats(model, dataset=None, task_metadata_path=None, save_path='embeddings'):
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    with open(task_metadata_path, 'r') as f:
        metadata = json.load(f)
    
    all_classes = []
    seen = set()
    
    for task_key in sorted(metadata.keys()): 
        known_classes = metadata[task_key]['known']
        for cls_name in known_classes:
            if cls_name not in seen:
                all_classes.append(cls_name)
                seen.add(cls_name)
    
    class_to_embedding = {}
    
    for cls_name in all_classes:
        feat = model.backbone.forward_text([cls_name])[0].squeeze(0).detach().cpu()  
        class_to_embedding[cls_name] = feat.squeeze(0) 
    
    if dataset is None:
        dataset = 'odinw13'
    
    dict_path = save_path / f'{dataset}_class_embeddings.pth'
    torch.save({
        'class_to_embedding': class_to_embedding,
        'all_classes': all_classes,
    }, dict_path)
    
    return class_to_embedding, all_classes

@torch.inference_mode()
def extract_feats_Co(model, dataset=None, task_metadata_path=None, save_path='embeddings', n_ctx=16, init_method='random'):
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    with open(task_metadata_path, 'r') as f:
        metadata = json.load(f)
    
    all_classes = []
    seen = set()
    
    for task_key in sorted(metadata.keys()): 
        known_classes = metadata[task_key]['known']
        for cls_name in known_classes:
            if cls_name not in seen:
                all_classes.append(cls_name)
                seen.add(cls_name)
    
    class_to_embedding = {}
    
    ctx_dim = 512

    for cls_name in all_classes:
        if init_method == 'text_based':
            raw_feat = model.backbone.forward_text([cls_name])[0].squeeze(0).detach().cpu() 
            
            if raw_feat.dim() == 1:
                embeddings = raw_feat.unsqueeze(0).repeat(n_ctx, 1) 
                embeddings += torch.randn_like(embeddings) * 0.02
            else:
                embeddings = raw_feat.repeat(n_ctx, 1)

        else: 
            embeddings = torch.empty(n_ctx, ctx_dim).normal_(mean=0.0, std=0.02)
        
        class_to_embedding[cls_name] = embeddings

    if dataset is None:
        dataset = 'odinw13'
    
    dict_path = save_path / f'{dataset}_class_Co_embeddings.pth'
    
    torch.save({
        'class_to_embedding': class_to_embedding, 
        'all_classes': all_classes,
        'n_ctx': n_ctx,
        'ctx_dim': ctx_dim
    }, dict_path)
    
    print(f"Saved embeddings to {dict_path}")
    return class_to_embedding, all_classes
    
def get_core_token_ids(text):

    if hasattr(model.backbone, 'tokenizer'):
        tokenizer = model.backbone.tokenizer
    elif hasattr(model.backbone, 'text_model') and hasattr(model.backbone.text_model, 'tokenizer'):
        tokenizer = model.backbone.text_model.tokenizer
    else:
        raise AttributeError("找不到 tokenizer，请在代码中手动指定 tokenizer 路径")

    full_ids = tokenizer(text)
    

    full_ids = full_ids['input_ids']
    
    if isinstance(full_ids, list):
        full_ids = torch.tensor(full_ids)
        
    if full_ids.dim() == 2:
        full_ids = full_ids[0] 

    valid_indices = torch.nonzero(full_ids).squeeze()
    if len(valid_indices) > 2:
        core_ids = full_ids[valid_indices[1:-1]]
    else:
        print(f"Warning: '{text}' tokenization weird, using random.")
        return None

    return core_ids
# 下面这个是用来调试到底embedding过不过clip有没有关系的
# @torch.inference_mode()
# def extract_feats_Co(model, dataset=None, task_metadata_path=None, save_path='embeddings', n_ctx=1, init_method='text_based'):
#     save_path = Path(save_path)
#     save_path.mkdir(parents=True, exist_ok=True)
    
#     with open(task_metadata_path, 'r') as f:
#         metadata = json.load(f)
    
#     all_classes = []
#     seen = set()
#     for task_key in sorted(metadata.keys()): 
#         known_classes = metadata[task_key]['known']
#         for cls_name in known_classes:
#             if cls_name not in seen:
#                 all_classes.append(cls_name)
#                 seen.add(cls_name)
    
#     class_to_embedding = {}
    
#     if hasattr(model.backbone, 'token_embedding'):
#         embedding_layer = model.backbone.token_embedding
#     elif hasattr(model.backbone, 'text_model'): 
#         try:
#             embedding_layer = model.backbone.text_model.embeddings.token_embedding
#         except AttributeError:
#             # 备用路径
#             embedding_layer = model.backbone.text_model.model.text_model.embeddings.token_embedding
#     else:
#         raise AttributeError("无法找到 token_embedding 层，请检查模型结构")

#     ctx_dim = embedding_layer.weight.shape[1] # 自动获取维度，通常是 512

   
#     for cls_name in all_classes:
#         if init_method == 'text_based':
#             try:
#                 core_ids = get_core_token_ids(cls_name)
                
#                 if core_ids is None:
#                     raise ValueError("Empty tokens")
                

#                 with torch.no_grad():
#                     raw_feat = embedding_layer(core_ids.to(embedding_layer.weight.device))
                
#                 if raw_feat.dim() == 2:
#                     raw_feat = raw_feat.mean(dim=0) # shape: [dim]
                
#                 raw_feat = raw_feat.float().cpu()

#             except Exception as e:
#                 print(f"Error initializing '{cls_name}': {e}. Fallback to random.")
#                 raw_feat = torch.randn(ctx_dim)

            
#             if raw_feat.dim() == 1:
#                 embeddings = raw_feat.unsqueeze(0).repeat(n_ctx, 1) 
#             else:
#                 embeddings = raw_feat.repeat(n_ctx, 1)

#         else: 
#             # 随机初始化
#             embeddings = torch.empty(n_ctx, ctx_dim).normal_(mean=0.0, std=0.02)
        
#         class_to_embedding[cls_name] = embeddings

#     if dataset is None:
#         dataset = 'odinw13'
    
#     dict_path = save_path / f'{dataset}_class_Co_embeddings.pth'
    
#     torch.save({
#         'class_to_embedding': class_to_embedding, 
#         'all_classes': all_classes,
#         'n_ctx': n_ctx,
#         'ctx_dim': ctx_dim
#     }, dict_path)
    
#     print(f"Saved embeddings to {dict_path}")
#     return class_to_embedding, all_classes
    
@torch.inference_mode()
def extract_tuned_feats(config=None, ckpt=None, wildcard='object', save_path="embeddings"):
    # extract tuned wildcard embeddings from text encoder
    model_path = sorted(glob.glob(f'work_dirs/{Path(config).stem}/best*.pth'))[-1] if ckpt is None else ckpt
    #model_path =  "pretrained/yolo_uniow_s_lora_bn_5e-4_100e_8gpus_obj365v1_goldg_train_lvis_minival.pth"
    tuned_feats = torch.load(model_path, map_location='cpu')['state_dict']['embeddings']
    print("Extracted tuned wildcard text features:", tuned_feats.shape)
    np.save(save_path / f'{wildcard.replace(" ", "_")}_tuned.npy', tuned_feats.numpy())


@torch.inference_mode()
def extract_wildcard_feats(model, wildcard='object', save_path='embeddings'):
    # extract wildcard embeddings from text encoder
    save_path = save_path / f'{wildcard.replace(" ", "_")}.npy'
    text_feats = model.backbone.forward_text([[wildcard]]).squeeze(0).detach().cpu()
    print("Extracted wildcard text features:", text_feats.shape)
    np.save(save_path, text_feats.numpy())


if __name__ == "__main__":
    args = parse_args()
    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    if args.extract_tuned:
        extract_tuned_feats(args.config, args.ckpt, save_path=save_path)
    else:
        # init model
        cfg = Config.fromfile(args.config)
        cfg.work_dir = 'work_dirs/extract_feats'

        runner = Runner.from_cfg(cfg)
        runner.call_hook("before_run")
        runner.load_checkpoint(args.ckpt, map_location='cpu')
        model = runner.model.to('cuda')
        model.eval()

        # extract features
        #extract_wildcard_feats(model, wildcard=args.wildcard, save_path=save_path)
        extract_feats(model, dataset='ZCOCO',task_metadata_path="/root/data-tmp/odinw13/ZCOCO_task_metadata.json", save_path=save_path)
        #extract_feats_Co(model, dataset='ODinW13',task_metadata_path="/root/data-tmp/odinw13/ODinW13_task_metadata.json", save_path=save_path)
        
       # for i in range(1, 14):
            
            #extract_feats(model, dataset='SOWODB', task=i, save_path=save_path)
            # if i < 4:
            #     extract_feats(model, dataset='nuOWODB', task=i, save_path=save_path)