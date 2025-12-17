import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch.utils.tensorboard import SummaryWriter
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.model import is_model_wrapper
import io
from PIL import Image

@HOOKS.register_module()
class TensorBoardLossHook(Hook):
    """记录各个loss到TensorBoard"""
    
    priority = 'NORMAL'
    
    def __init__(self, task_id="", log_interval=10):
        self.log_interval = log_interval
        self.writer = None
        self.task_id = task_id
    
    def before_train(self, runner):
        self.log_dir = os.path.join(runner.cfg.get("work_dir"),"tensorboard",self.task_id)
        self.writer = SummaryWriter(log_dir=self.log_dir)
    
    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        """记录loss"""
        if runner.iter % self.log_interval != 0:
            return
        
        # 从outputs中提取loss
        if outputs is not None and 'loss' in outputs:
            losses = outputs['loss']
            
            # 记录总loss
            if isinstance(losses, dict):
                for key, value in losses.items():
                    if torch.is_tensor(value):
                        self.writer.add_scalar(f'Loss/{key}', 
                                             value.item(), runner.iter)
    
    def after_train(self, runner):
        if self.writer is not None:
            self.writer.close()