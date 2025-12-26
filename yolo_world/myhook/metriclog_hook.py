import os
import json
import numpy as np
import torch
import torch.distributed as dist
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.logging import MMLogger
from mmengine.model import is_model_wrapper

def is_main():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


@HOOKS.register_module()
class OWODMetricHook(Hook):
    def __init__(self, 
                 json_path='work_dirs/owod_metrics.json', 
                 metric_key='owod/CK',
                 task_id=1,
                 save_best=True,
                 save_embeddings=False): 
        self.task_id = task_id
        self.json_path = json_path
        self.metric_key = metric_key
        self.save_best = save_best
        self.save_embeddings = save_embeddings
        self.logger = MMLogger.get_current_instance()

    def _save_current_embeddings(self, runner):
        if not is_main():
            return  # 只让 rank0 保存文件

        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        
        cur_embeddings = model.get_cur_embeddings()
        cur_classes = model.get_cls_names()
        prev_embeddings = model.get_prev_embeddings()
        prev_classes = model.get_prev_cls_names()

        save_dir = os.path.join("/".join(self.json_path.split("/")[:-1]),
                                f"t{self.task_id}_valid_text_embeddings")
        os.makedirs(save_dir, exist_ok=True)

        epoch = runner.epoch
        filename = os.path.join(save_dir, f'embeddings_epoch_{epoch}.npy')

        save_data = {
            'epoch': epoch,
            #'cur_embeddings': cur_embeddings.cpu().numpy(),
            'cur_classes': cur_classes,
            #'prev_embeddings': prev_embeddings.cpu().numpy() if prev_embeddings is not None else None,
            'prev_classes': prev_classes,
        }

        np.save(filename, save_data)
        self.logger.info(f"[OWODHook] Saved text embeddings to {filename}")

    def _update_record(self, metrics):
        if not is_main():
            return  # 只让 rank0 写 JSON

        if not metrics:
            return
        
        current_map = None

        # 查找 mAP
        if self.metric_key in metrics:
            current_map = metrics[self.metric_key]
        else:
            for k in metrics:
                if 'mAP' in k and '50' not in k and '75' not in k:
                    current_map = metrics[k]
                    break
        if current_map is None:
            return

        current_map = float(current_map)
        task_key = f"task_{self.task_id}"

        # 加载已有 JSON
        record = {}
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, 'r') as f:
                    record = json.load(f)
            except json.JSONDecodeError:
                pass

        updated = False
        if self.save_best and task_key in record:
            old_map = record[task_key]
            if current_map > old_map:
                record[task_key] = current_map
                self.logger.info(f"Task {self.task_id} New Best mAP: {current_map:.4f}")
                updated = True
        else:
            record[task_key] = current_map
            self.logger.info(f"Task {self.task_id} mAP: {current_map:.4f}")
            updated = True

        if updated:
            os.makedirs(os.path.dirname(self.json_path), exist_ok=True)
            with open(self.json_path, 'w') as f:
                json.dump(record, f, indent=4)

    def after_val_epoch(self, runner, metrics):
        if self.save_embeddings:
            self._save_current_embeddings(runner)
        self._update_record(metrics)

    def after_test_epoch(self, runner, metrics):
        self._update_record(metrics)