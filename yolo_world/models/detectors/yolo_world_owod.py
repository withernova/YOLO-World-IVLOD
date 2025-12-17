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
from .CoOp_injector import CLIPEmbeddingHook

@MODELS.register_module()
class OWODDetector(YOLODetector):
    """Implementation of Open-World YOLO (Dict Mode Only) with YOLO-World Adapter"""

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
                 **kwargs) -> None:
        self.mode = mode
        self.mm_neck = mm_neck
        self.prompt_dim = prompt_dim
        self.freeze_prompt = freeze_prompt
        self.use_mlp_adapter = use_mlp_adapter
        self.task_id = task_id
        
        super().__init__(*args, **kwargs)

        with open(task_metadata_path, 'r') as f:
            self.task_metadata = json.load(f)
        
        self.all_class_embeddings_path = all_class_embeddings_path

        if os.path.exists(all_class_embeddings_path):
            checkpoint = torch.load(all_class_embeddings_path, map_location='cpu')
            class_to_embedding = checkpoint['class_to_embedding'] 
        else:
            # Fallback for init or first task if file missing
            class_to_embedding = {} 
        
        task_key = f"t{task_id}"
        known_classes = self.task_metadata[task_key]["known"] 
        cur_known_classes = self.task_metadata[task_key]["task_classes"]  
        
        prev_classes = [cls for cls in known_classes if cls not in cur_known_classes]
        ordered_classes = prev_classes + cur_known_classes
        
        emb_list = []
        for cls in ordered_classes:
            if cls in class_to_embedding:
                emb_list.append(class_to_embedding[cls])
            else:
                # 简单的随机初始化 Fallback，防止 crash
                emb_list.append(torch.randn(prompt_dim))
        
        known_embeddings = torch.stack(emb_list)
        
        self.embeddings = nn.Parameter(known_embeddings.float())

        if mode != 'train' and os.path.exists(mode):
            print(f"🔄 [OWODDetector] Mode is '{mode}', loading embeddings from file...")
            self._update_test(ordered_classes=ordered_classes)
        
        self.num_prev_classes = len(prev_classes)
        self.num_cur_classes = len(cur_known_classes)
        self.num_known_classes = len(ordered_classes)  # prev + cur
        
        self.num_training_classes = self.num_known_classes
        self.num_test_classes = self.num_training_classes
        
        self.num_total_embeddings = known_embeddings.shape[0]
        
        self.ordered_classes = ordered_classes
        self.cur_known_classes = cur_known_classes
        self.prev_classes = prev_classes
        
        # --- 保持原有的梯度 Mask 逻辑 ---
        self._grad_mask = torch.zeros(self.num_total_embeddings, dtype=torch.bool)[:,None, None]

        if len(self.embeddings.shape) == 2:
             self._grad_mask = torch.zeros(self.num_total_embeddings, dtype=torch.bool)[:, None]

        cur_start_idx = self.num_prev_classes
        cur_end_idx = self.num_prev_classes + self.num_cur_classes
        self._grad_mask[cur_start_idx:cur_end_idx] = True
        
        self.embeddings.requires_grad = True
        self.embeddings.register_hook(lambda grad: grad * self._grad_mask.to(grad.device))

        if use_mlp_adapter:
            self.adapter = nn.Sequential(
                nn.Linear(prompt_dim, prompt_dim * 2),
                nn.ReLU(True),
                nn.Linear(prompt_dim * 2, prompt_dim)
            )
        else:
            self.adapter = None
            
        # Hook 注入
        self.hook = CLIPEmbeddingHook(
            clip_model=self.backbone.text_model,
        )
            
    def _update_test(self, ordered_classes):
        """保持原有逻辑"""
        external_checkpoint = torch.load(self.mode, map_location='cpu')
        
        if 'class_to_embedding' in external_checkpoint:
            external_dict = external_checkpoint['class_to_embedding']
        elif isinstance(external_checkpoint, dict):
            external_dict = external_checkpoint

        update_count = 0
        with torch.no_grad():
            for idx, cls_name in enumerate(ordered_classes):
                if cls_name in external_dict:
                    new_emb = external_dict[cls_name]
                    
                    if not isinstance(new_emb, torch.Tensor):
                        new_emb = torch.tensor(new_emb)
                    new_emb = new_emb.to(self.embeddings.device).float()
                    
                    if new_emb.shape == self.embeddings[idx].shape:
                        self.embeddings[idx].copy_(new_emb)
                        update_count += 1
        
    def loss(self, batch_inputs: Tensor, batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses."""
        self.bbox_head.num_classes = self.num_training_classes
        
        img_feats, txt_feats = self.extract_feat(batch_inputs, batch_data_samples)
        batch_size, num_classes, _ = txt_feats.shape
        txt_masks = txt_feats.new_ones((batch_size, num_classes), dtype=torch.long)
        losses = self.bbox_head.loss(img_feats, txt_feats, txt_masks, batch_data_samples)
        return losses

    def predict(self, batch_inputs: Tensor, batch_data_samples: SampleList, 
                rescale: bool = True) -> SampleList:
        """Predict results."""
        img_feats, txt_feats = self.extract_feat(batch_inputs, batch_data_samples)

        # 动态设置类别数，基于当前的 txt_feats
        # txt_feats shape: (B, Num_Classes, Dim)
        self.bbox_head.num_classes = txt_feats.shape[1]
        batch_size, num_classes, _ = txt_feats.shape
        txt_masks = txt_feats.new_ones((batch_size, num_classes), dtype=torch.long)
        results_list = self.bbox_head.predict(img_feats, txt_feats, txt_masks,
                                               batch_data_samples, rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(batch_data_samples, results_list)
        return batch_data_samples

    def _forward(self, batch_inputs: Tensor, 
                 batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process."""
        img_feats, txt_feats = self.extract_feat(batch_inputs, batch_data_samples)
        batch_size, num_classes, _ = txt_feats.shape
        txt_masks = txt_feats.new_ones((batch_size, num_classes), dtype=torch.long)
        
        results = self.bbox_head.forward(img_feats, txt_feats, txt_masks)
        return results

    def extract_feat(self, batch_inputs: Tensor, 
                     batch_data_samples: SampleList) -> Tuple[Tuple[Tensor], Tensor]:
        """Extract features."""
        B = batch_inputs.shape[0]
        cur_embeddings = self.embeddings[:self.num_training_classes] # n_ctx个可训练的embedding 
        self.hook.set_embedding(cur_embeddings)
        img_feats, _ = self.backbone(batch_inputs, None)
        class_names = self.ordered_classes[:self.num_training_classes]
        #class_names = ["a"] * self.num_training_classes
        with self.hook:
            txt_feats = self.backbone.forward_text([class_names])[0]
            #txt_feats = self.backbone.forward_text([" "])
            
            
        #txt_feats = self.embeddings[:self.num_training_classes]
        
        if txt_feats.shape[0] == 1:
            txt_feats = txt_feats.repeat(img_feats[0].shape[0], 1, 1)
        
        if self.adapter is not None:
            txt_feats = self.adapter(txt_feats) + txt_feats
            txt_feats = nn.functional.normalize(txt_feats, dim=-1, p=2)
        
        #txt_feats = txt_feats.repeat(img_feats[0].shape[0], 1, 1)

        if self.with_neck:
            if self.mm_neck:
                img_feats = self.neck(img_feats, txt_feats)
            else:
                img_feats = self.neck(img_feats)
        
        return img_feats, txt_feats


    def update_class_embeddings_dict(self):
        if dist.is_initialized() and dist.get_rank() != 0:
            dist.barrier()  
            return

        cur_start_idx = self.num_prev_classes
        cur_end_idx = cur_start_idx + self.num_cur_classes
        updated_cur_embeddings = self.embeddings[cur_start_idx:cur_end_idx].detach().cpu()

        checkpoint = torch.load(self.all_class_embeddings_path, map_location='cpu')
        class_to_embedding = checkpoint['class_to_embedding']

        for i, cls_name in enumerate(self.cur_known_classes):
            class_to_embedding[cls_name] = updated_cur_embeddings[i]

        torch.save(
            {'class_to_embedding': class_to_embedding},
            self.all_class_embeddings_path
        )

        if dist.is_initialized():
            dist.barrier()

    def get_cur_embeddings(self):
        cur_start_idx = self.num_prev_classes
        cur_end_idx = self.num_prev_classes + self.num_cur_classes
        return self.embeddings[cur_start_idx:cur_end_idx].detach().cpu()
        
    def get_cls_names(self):
        return self.cur_known_classes

    def get_prev_cls_names(self):
        return self.prev_classes

    def get_all_cls_names(self):
        return self.ordered_classes

    def get_cls_embedding_by_name(self, cls_name):
        if cls_name in self.ordered_classes:
            idx = self.ordered_classes.index(cls_name)
            return self.embeddings[idx].detach()
        return None

    def compute_gradient_conflict(self, loss_dict):
        gradients = {}
        for class_name, loss in loss_dict.items():
            if self.embeddings.grad is not None:
                self.embeddings.grad.zero_()
            
            loss.backward(retain_graph=True)
            
            if self.embeddings.grad is not None:
                cur_start_idx = self.num_prev_classes
                cur_end_idx = self.num_prev_classes + self.num_cur_classes
                grad_vector = self.embeddings.grad[cur_start_idx:cur_end_idx].flatten().clone()
                gradients[class_name] = grad_vector
        
        n_classes = len(gradients)
        similarity_matrix = torch.zeros((n_classes, n_classes))
        class_list = list(gradients.keys())
        for i, class_i in enumerate(class_list):
            for j, class_j in enumerate(class_list):
                if i != j:
                    cos_sim = torch.nn.functional.cosine_similarity(
                        gradients[class_i].unsqueeze(0),
                        gradients[class_j].unsqueeze(0)
                    )
                    similarity_matrix[i, j] = cos_sim.item()
                else:
                    similarity_matrix[i, j] = 1.0
        return similarity_matrix.cpu().numpy(), class_list

    def get_embedding_stats(self):
        cur_start_idx = self.num_prev_classes
        cur_end_idx = self.num_prev_classes + self.num_cur_classes
        
        cur_embeddings = self.embeddings[cur_start_idx:cur_end_idx]
        prev_embeddings = self.embeddings[:self.num_prev_classes] if self.num_prev_classes > 0 else None
        
        stats = {
            'cur_norm_mean': cur_embeddings.norm(dim=-1).mean().item(),
            'cur_norm_std': cur_embeddings.norm(dim=-1).std().item(),
        }
        
        if prev_embeddings is not None:
            stats['prev_norm_mean'] = prev_embeddings.norm(dim=-1).mean().item()
            stats['prev_norm_std'] = prev_embeddings.norm(dim=-1).std().item()
            
            similarity = torch.mm(
                torch.nn.functional.normalize(cur_embeddings, dim=-1),
                torch.nn.functional.normalize(prev_embeddings, dim=-1).t()
            )
            stats['inter_similarity_mean'] = similarity.mean().item()
            stats['inter_similarity_max'] = similarity.max().item()
        
        return stats
    
    def get_cur_embeddings_grad(self):
        if self.embeddings.grad is None:
            return None
        cur_start_idx = self.num_prev_classes
        cur_end_idx = self.num_prev_classes + self.num_cur_classes
        return self.embeddings.grad[cur_start_idx:cur_end_idx].detach().cpu()

    def get_prev_embeddings(self):
        if self.num_prev_classes == 0:
            return None
        return self.embeddings[:self.num_prev_classes].detach().cpu()

    def get_all_embeddings(self):
        return self.embeddings[:self.num_known_classes].detach().cpu()

    def get_per_class_grad_norms(self):
        if self.embeddings.grad is None:
            return None
        cur_start_idx = self.num_prev_classes
        cur_end_idx = self.num_prev_classes + self.num_cur_classes
        cur_grad = self.embeddings.grad[cur_start_idx:cur_end_idx]
        per_class_norms = torch.norm(cur_grad, dim=-1)
        return per_class_norms.detach().cpu()