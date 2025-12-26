# Copyright (c) Tencent Inc. All rights reserved.
from typing import List, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from mmdet.structures import OptSampleList, SampleList
from mmyolo.models.detectors import YOLODetector
from mmyolo.registry import MODELS
import json
import os
import torch.distributed as dist
# 确保引用的是上面修改过的 Hook
from .CoOp_injector import CLIPEmbeddingHook 

@MODELS.register_module()
class OWODDetector(YOLODetector):
    """Implementation of Open-World YOLO with Zero-Shot Placeholder Support"""

    def __init__(self,
                 *args,
                 mm_neck: bool = True, 
                 prompt_dim: int = 512,
                 freeze_prompt: bool = False,
                 use_mlp_adapter: bool = False,
                 task_metadata_path: str = '',
                 all_class_embeddings_path: str = '',
                 task_id: int = 1,
                 mode: str = 'train',
                 n_ctx=16,
                 **kwargs) -> None:
        self.mode = mode
        self.mm_neck = mm_neck
        self.prompt_dim = prompt_dim
        self.freeze_prompt = freeze_prompt
        self.use_mlp_adapter = use_mlp_adapter
        self.task_id = task_id
        self.n_ctx = n_ctx
        
        super().__init__(*args, **kwargs)

        with open(task_metadata_path, 'r') as f:
            self.task_metadata = json.load(f)
        
        self.all_class_embeddings_path = all_class_embeddings_path

        if os.path.exists(all_class_embeddings_path):
            checkpoint = torch.load(all_class_embeddings_path, map_location='cpu')
            class_to_embedding = checkpoint['class_to_embedding'] 
        else:
            class_to_embedding = {} 
        
        task_key = f"t{task_id}"
        known_classes = self.task_metadata[task_key]["known"] 
        cur_known_classes = self.task_metadata[task_key]["task_classes"]  
        
        prev_classes = [cls for cls in known_classes if cls not in cur_known_classes]
        ordered_classes = prev_classes + cur_known_classes
        
        self.num_prev_classes = len(prev_classes)
        self.num_cur_classes = len(cur_known_classes)
        self.num_known_classes = len(ordered_classes)
        self.num_training_classes = self.num_known_classes
        
        self.invalid_class_names = set()

        prev_params = []
        for cls in prev_classes:
            if cls in class_to_embedding:
                emb = class_to_embedding[cls]
                if emb.dim() == 1: 
                    emb = emb.unsqueeze(0)
            else:
                emb = torch.randn(self.n_ctx, prompt_dim)
            
            p = nn.Parameter(emb.float())
            p.requires_grad = False 
            prev_params.append(p)
            
        self.prev_embeddings = nn.ParameterList(prev_params)

        cur_params = []
        for cls in cur_known_classes:
            if cls in class_to_embedding:
                emb = class_to_embedding[cls]
                if emb.dim() == 1: emb = emb.unsqueeze(0)
            else:
                emb = torch.randn(self.n_ctx, prompt_dim)
            
            p = nn.Parameter(emb.float())
            p.requires_grad = True 
            cur_params.append(p)

        self.cur_embeddings = nn.ParameterList(cur_params)
        self.ordered_classes = ordered_classes
        self.cur_known_classes = cur_known_classes
        self.prev_classes = prev_classes

        # 如果是测试模式，尝试加载 update
        if mode != 'train' and os.path.exists(mode):
            print(f"🔄 [OWODDetector] Mode is '{mode}', loading embeddings from file...")
            self._update_test(ordered_classes=ordered_classes)
        
        if use_mlp_adapter:
            self.adapter = nn.Sequential(
                nn.Linear(prompt_dim, prompt_dim * 2),
                nn.ReLU(True),
                nn.Linear(prompt_dim * 2, prompt_dim)
            )
        else:
            self.adapter = None
            
        self.hook = CLIPEmbeddingHook(
            clip_model=self.backbone.text_model,
        )

    @property
    def embeddings(self):
        return list(self.prev_embeddings) + list(self.cur_embeddings)
            
    def _update_test(self, ordered_classes):
        
        print(f"🔄 [OWODDetector] Loading embeddings from {self.mode} ...")
        external_checkpoint = torch.load(self.mode, map_location='cpu')
        
        if 'class_to_embedding' in external_checkpoint:
            external_dict = external_checkpoint['class_to_embedding']
        elif isinstance(external_checkpoint, dict):
            external_dict = external_checkpoint

        def update_param_list(param_list, class_names):
            for idx, cls_name in enumerate(class_names):
                self.invalid_class_names.add(cls_name)
                
                # if cls_name in external_dict:
                #     new_emb = external_dict[cls_name]
                #     if new_emb is None: continue 

                #     if not isinstance(new_emb, torch.Tensor):
                #         new_emb = torch.tensor(new_emb)
                    
                #     device = param_list[idx].device
                #     new_emb = new_emb.to(device).float()
                    
                #     if new_emb.dim() == 1:
                #         new_emb = new_emb.unsqueeze(0)

                #     original_shape = param_list[idx].shape
                #     if new_emb.shape != original_shape:
                #         new_param = nn.Parameter(new_emb)
                #         new_param.requires_grad = param_list[idx].requires_grad
                        
                #         param_list[idx] = new_param
                #     else:
                #         with torch.no_grad():
                #             param_list[idx].data.copy_(new_emb)
                # else:
                #     self.invalid_class_names.add(cls_name)

        if self.prev_classes is not None:
            update_param_list(self.prev_embeddings, self.prev_classes)
        update_param_list(self.cur_embeddings, self.cur_known_classes)
        
    def extract_feat(self, batch_inputs: Tensor, 
                     batch_data_samples: SampleList) -> Tuple[Tuple[Tensor], Tensor]:
        """Extract features."""
        
        full_embeddings_raw = self.embeddings # 这是一个全量的 Tensor List
        
        hook_input_list = []
        for cls_name, emb_tensor in zip(self.ordered_classes, full_embeddings_raw):
            if cls_name in self.invalid_class_names:
                hook_input_list.append(None)
            else:
                hook_input_list.append(emb_tensor)
        
        self.hook.set_embedding(hook_input_list)
        
        img_feats, _ = self.backbone(batch_inputs, None)
        class_names = self.ordered_classes[:self.num_training_classes]

        with self.hook:
            txt_feats = self.backbone.forward_text([class_names])[0]
        
        if txt_feats.dim() == 2:
            txt_feats = txt_feats.unsqueeze(0)
            
        if txt_feats.shape[0] == 1:
            txt_feats = txt_feats.repeat(img_feats[0].shape[0], 1, 1)
            
        if self.adapter is not None:
            txt_feats = self.adapter(txt_feats) + txt_feats
        txt_feats = nn.functional.normalize(txt_feats, dim=-1, p=2)
        
        if self.with_neck:
            if self.mm_neck:
                img_feats = self.neck(img_feats, txt_feats)
            else:
                img_feats = self.neck(img_feats)
        
        return img_feats, txt_feats

    def loss(self, batch_inputs: Tensor, batch_data_samples: SampleList) -> Union[dict, list]:
        self.bbox_head.num_classes = self.num_training_classes
        img_feats, txt_feats = self.extract_feat(batch_inputs, batch_data_samples)
        batch_size, num_classes, _ = txt_feats.shape
        txt_masks = txt_feats.new_ones((batch_size, num_classes), dtype=torch.long)
        losses = self.bbox_head.loss(img_feats, txt_feats, txt_masks, batch_data_samples)
        return losses

    def predict(self, batch_inputs: Tensor, batch_data_samples: SampleList, rescale: bool = True) -> SampleList:
        img_feats, txt_feats = self.extract_feat(batch_inputs, batch_data_samples)
        self.bbox_head.num_classes = txt_feats.shape[1]
        batch_size, num_classes, _ = txt_feats.shape
        txt_masks = txt_feats.new_ones((batch_size, num_classes), dtype=torch.long)
        results_list = self.bbox_head.predict(img_feats, txt_feats, txt_masks, batch_data_samples, rescale=rescale)
        batch_data_samples = self.add_pred_to_datasample(batch_data_samples, results_list)
        return batch_data_samples

    def _forward(self, batch_inputs: Tensor, batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        img_feats, txt_feats = self.extract_feat(batch_inputs, batch_data_samples)
        batch_size, num_classes, _ = txt_feats.shape
        txt_masks = txt_feats.new_ones((batch_size, num_classes), dtype=torch.long)
        results = self.bbox_head.forward(img_feats, txt_feats, txt_masks)
        return results

    def update_class_embeddings_dict(self):
        if dist.is_initialized() and dist.get_rank() != 0:
            dist.barrier()  
            return
        updated_cur_embeddings = [p.detach().cpu() for p in self.cur_embeddings]
        checkpoint = torch.load(self.all_class_embeddings_path, map_location='cpu')
        class_to_embedding = checkpoint['class_to_embedding']
        for i, cls_name in enumerate(self.cur_known_classes):
            class_to_embedding[cls_name] = updated_cur_embeddings[i]
        torch.save({'class_to_embedding': class_to_embedding}, self.all_class_embeddings_path)
        if dist.is_initialized():
            dist.barrier()

    def get_cur_embeddings(self):
        return [p.detach().cpu() for p in self.cur_embeddings] # Return list to be safe
        
    def get_cls_names(self):
        return self.cur_known_classes

    def get_prev_cls_names(self):
        return self.prev_classes

    def get_all_cls_names(self):
        return self.ordered_classes

    def get_cls_embedding_by_name(self, cls_name):
        full_emb = self.embeddings
        if cls_name in self.ordered_classes:
            idx = self.ordered_classes.index(cls_name)
            return full_emb[idx].detach()
        return None
    
    def get_cur_embeddings_grad(self):
        grads = []
        for p in self.cur_embeddings:
            if p.grad is not None:
                grads.append(p.grad.detach().cpu())
            else:
                grads.append(None)
        return grads

    def get_prev_embeddings(self):
        if self.prev_embeddings is None: return None
        return [p.detach().cpu() for p in self.prev_embeddings]

    def get_all_embeddings(self):
        return [p.detach().cpu() for p in self.embeddings]

    def get_per_class_grad_norms(self):
        norms = []
        for p in self.cur_embeddings:
            if p.grad is not None:
                norms.append(p.grad.norm().item())
            else:
                norms.append(0.0)
        return torch.tensor(norms)