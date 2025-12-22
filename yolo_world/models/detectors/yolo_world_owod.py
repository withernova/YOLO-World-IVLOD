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
        
        emb_list = []
        for cls in ordered_classes:
            if cls in class_to_embedding:
                emb_list.append(class_to_embedding[cls])
            else:
                emb_list.append(torch.randn(self.n_ctx, prompt_dim)) 
        
        known_embeddings = torch.stack(emb_list)

        if self.num_prev_classes > 0:
            self.prev_embeddings = nn.Parameter(known_embeddings[:self.num_prev_classes].float())
            self.prev_embeddings.requires_grad = False 
        else:
            self.register_parameter('prev_embeddings', None)
        self.cur_embeddings = nn.Parameter(known_embeddings[self.num_prev_classes:].float())
        self.cur_embeddings.requires_grad = True

        if mode != 'train' and os.path.exists(mode):
            print(f"🔄 [OWODDetector] Mode is '{mode}', loading embeddings from file...")
            self._update_test(ordered_classes=ordered_classes)
        
        self.ordered_classes = ordered_classes
        self.cur_known_classes = cur_known_classes
        self.prev_classes = prev_classes
        
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
            n_ctx=self.n_ctx
        )

    @property
    def embeddings(self):
        if self.prev_embeddings is not None:
            return torch.cat([self.prev_embeddings, self.cur_embeddings], dim=0)
        return self.cur_embeddings
            
    def _update_test(self, ordered_classes):
        external_checkpoint = torch.load(self.mode, map_location='cpu')
        
        if 'class_to_embedding' in external_checkpoint:
            external_dict = external_checkpoint['class_to_embedding']
        elif isinstance(external_checkpoint, dict):
            external_dict = external_checkpoint

        # 获取完整的临时 tensor 用于更新
        full_embeddings = self.embeddings.clone()

        with torch.no_grad():
            for idx, cls_name in enumerate(ordered_classes):
                if cls_name in external_dict:
                    new_emb = external_dict[cls_name]
                    if not isinstance(new_emb, torch.Tensor):
                        new_emb = torch.tensor(new_emb)
                    new_emb = new_emb.to(full_embeddings.device).float()
                    
                    if new_emb.dim() == 1:
                        new_emb = new_emb.unsqueeze(0)

                    if new_emb.shape == full_embeddings[idx].shape:
                        full_embeddings[idx].copy_(new_emb)

        # 将更新后的值写回 Parameter
        if self.prev_embeddings is not None:
            self.prev_embeddings.data.copy_(full_embeddings[:self.num_prev_classes])
        self.cur_embeddings.data.copy_(full_embeddings[self.num_prev_classes:])
        
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
        # 这里使用 property self.embeddings 自动获取拼接后的 Tensor
        # 此时 self.embeddings 包含了 prev (frozen) + cur (trainable)
        full_embeddings = self.embeddings 
        
        # 注入 Hook
        self.hook.set_embedding(full_embeddings)
        
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

    def update_class_embeddings_dict(self):
        if dist.is_initialized() and dist.get_rank() != 0:
            dist.barrier()  
            return

        # 直接使用 cur_embeddings，无需切片
        updated_cur_embeddings = self.cur_embeddings.detach().cpu()

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
        # 直接返回 cur_embeddings
        return self.cur_embeddings.detach().cpu()
        
    def get_cls_names(self):
        return self.cur_known_classes

    def get_prev_cls_names(self):
        return self.prev_classes

    def get_all_cls_names(self):
        return self.ordered_classes

    def get_cls_embedding_by_name(self, cls_name):
        # 需要从拼接后的 embeddings 中查找
        full_emb = self.embeddings
        if cls_name in self.ordered_classes:
            idx = self.ordered_classes.index(cls_name)
            return full_emb[idx].detach()
        return None

    def compute_gradient_conflict(self, loss_dict):
        # 注意：这里只计算 cur_embeddings 的梯度冲突，因为 prev 没有梯度
        gradients = {}
        for class_name, loss in loss_dict.items():
            if self.cur_embeddings.grad is not None:
                self.cur_embeddings.grad.zero_()
            
            loss.backward(retain_graph=True)
            
            if self.cur_embeddings.grad is not None:
                # 假设 class_name 属于当前任务，找到它在 cur_embeddings 中的相对索引
                if class_name in self.cur_known_classes:
                    rel_idx = self.cur_known_classes.index(class_name)
                    grad_vector = self.cur_embeddings.grad[rel_idx].flatten().clone()
                    gradients[class_name] = grad_vector
        
        if not gradients:
            return np.zeros((0, 0)), []

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
        # 直接使用分开的变量
        cur_embeddings = self.cur_embeddings
        prev_embeddings = self.prev_embeddings
        
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
        if self.cur_embeddings.grad is None:
            return None
        return self.cur_embeddings.grad.detach().cpu()

    def get_prev_embeddings(self):
        if self.prev_embeddings is None:
            return None
        return self.prev_embeddings.detach().cpu()

    def get_all_embeddings(self):
        return self.embeddings.detach().cpu()

    def get_per_class_grad_norms(self):
        if self.cur_embeddings.grad is None:
            return None
        per_class_norms = torch.norm(self.cur_embeddings.grad, dim=-1)
        return per_class_norms.detach().cpu()