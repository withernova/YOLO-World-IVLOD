import os
import torch
import pickle
import numpy as np
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.model import is_model_wrapper

@HOOKS.register_module()
class EmbeddingTrajectoryHook(Hook):
    """记录embedding轨迹用于后续TSNE可视化"""
    
    priority = 'NORMAL'
    
    def __init__(self, 
                 task_id="task1",
                 save_interval=1,
                 use_tensorboard=True):
        self.task_id = task_id
        self.save_interval = save_interval
        self.use_tensorboard = use_tensorboard
        
        # 核心数据
        self.data = {
            'metadata': {
                'task_id': task_id,
                'cur_class_names': [],    # 新增类别
                'prev_class_names': [],   # 之前类别
                'all_class_names': [],    # 所有类别
                'num_prev_classes': 0,
                'num_cur_classes': 0,
            },
            # ⭐ 完整的embedding历史
            'embeddings': {
                'epochs': [],
                'cur_embeddings': [],   # [epoch, num_cur, dim]
                'prev_embeddings': None,  # [num_prev, dim] 只保存一次
                'all_embeddings': [],   # [epoch, num_all, dim]
            },
            # 梯度历史
            'gradients': {
                'epochs': [],
                'avg_grad_norm': [],           # 总体平均
                'per_class_grad_norms': [],    # [epoch, num_cur]
            },
            # 性能指标
            'metrics': {
                'epochs': [],
                'ap50': [],
            }
        }
        
        self.current_epoch_grad_norms = []
        self.current_epoch_per_class_grads = []
        
    def before_train(self, runner):
        """训练开始前初始化"""
        self.work_dir = runner.cfg.get("work_dir")
        self.save_dir = os.path.join(self.work_dir, "embedding_analysis")
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.data_file = os.path.join(self.save_dir, f"{self.task_id}_trajectory.pkl")
        
        # TensorBoard writer
        if self.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.tb_writer = SummaryWriter(
                log_dir=os.path.join(self.work_dir, "tensorboard", self.task_id)
            )
        else:
            self.tb_writer = None
        
        print(f"📊 Embedding轨迹Hook已启动")
        print(f"   保存目录: {self.save_dir}")
        print(f"   数据文件: {self.data_file}")
        
    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        """记录梯度"""
        if is_model_wrapper(runner.model):
            model = runner.model.module
        else:
            model = runner.model
        
        # 1. 总体梯度norm
        cur_grad = model.get_cur_embeddings_grad()
        if cur_grad is not None:
            grad_norm = cur_grad.norm().item()
            self.current_epoch_grad_norms.append(grad_norm)
        
        # 2. 每个类别的梯度norm
        per_class_norms = model.get_per_class_grad_norms()
        if per_class_norms is not None:
            self.current_epoch_per_class_grads.append(per_class_norms.numpy())
    
    def after_val_epoch(self, runner, metrics=None):
        """验证后获取AP50指标"""
        if metrics is None and hasattr(runner, 'message_hub'):
            try:
                val_info = runner.message_hub.get_info('val')
                if val_info is not None and isinstance(val_info, dict):
                    metrics = val_info
            except:
                pass
        
        # 提取AP50
        ap50 = None
        if metrics is not None and isinstance(metrics, dict):
            for key in ['owod/Both', 'Both', 'mAP50', 'AP50']:
                if key in metrics:
                    ap50 = metrics[key]
                    break
        
        self._last_ap50 = ap50
    
    def after_train_epoch(self, runner):
        """每个epoch后保存数据"""
        if is_model_wrapper(runner.model):
            model = runner.model.module
        else:
            model = runner.model
        
        epoch = runner.epoch
        
        # ⭐ 1. 保存完整的embeddings
        cur_embeddings = model.get_cur_embeddings()  # [num_cur, dim]
        all_embeddings = model.get_all_embeddings()  # [num_all, dim]
        
        self.data['embeddings']['epochs'].append(epoch)
        self.data['embeddings']['cur_embeddings'].append(cur_embeddings.numpy())
        self.data['embeddings']['all_embeddings'].append(all_embeddings.numpy())
        
        # 第一次保存prev_embeddings
        if epoch == 0:
            prev_embeddings = model.get_prev_embeddings()
            if prev_embeddings is not None:
                self.data['embeddings']['prev_embeddings'] = prev_embeddings.numpy()
        
        # 2. 梯度统计
        if self.current_epoch_grad_norms:
            avg_grad_norm = sum(self.current_epoch_grad_norms) / len(self.current_epoch_grad_norms)
        else:
            avg_grad_norm = 0.0
        
        if self.current_epoch_per_class_grads:
            avg_per_class_grads = np.mean(self.current_epoch_per_class_grads, axis=0)
        else:
            avg_per_class_grads = np.zeros(model.num_cur_classes)
        
        self.data['gradients']['epochs'].append(epoch)
        self.data['gradients']['avg_grad_norm'].append(avg_grad_norm)
        self.data['gradients']['per_class_grad_norms'].append(avg_per_class_grads)
        
        # 3. AP50
        ap50 = getattr(self, '_last_ap50', None)
        self.data['metrics']['epochs'].append(epoch)
        self.data['metrics']['ap50'].append(ap50)
        
        # 4. TensorBoard可视化
        if self.tb_writer is not None:
            self.tb_writer.add_scalar('Train/GradientNorm', avg_grad_norm, epoch)
            
            if ap50 is not None:
                self.tb_writer.add_scalar('Val/AP50', ap50, epoch)
            
            # 每个类别的梯度
            cur_class_names = model.get_cls_names()
            for i, class_name in enumerate(cur_class_names):
                self.tb_writer.add_scalar(f'Gradient/Norm_{class_name}', 
                                        avg_per_class_grads[i], epoch)
            
            # ⭐ Embedding可视化 (每5个epoch)
            if epoch % 5 == 0 and len(all_embeddings) > 1:
                try:
                    all_class_names = model.get_all_cls_names()
                    self.tb_writer.add_embedding(
                        all_embeddings,
                        metadata=all_class_names,
                        global_step=epoch,
                        tag='all_embeddings'
                    )
                except Exception as e:
                    print(f"⚠️ TensorBoard embedding可视化失败: {e}")
        
        # 5. 更新metadata (第一次)
        if epoch == 0:
            self.data['metadata']['cur_class_names'] = model.get_cls_names()
            self.data['metadata']['prev_class_names'] = model.get_prev_cls_names()
            self.data['metadata']['all_class_names'] = model.get_all_cls_names()
            self.data['metadata']['num_prev_classes'] = model.num_prev_classes
            self.data['metadata']['num_cur_classes'] = model.num_cur_classes
        
        # 6. 定期保存和打印
        if (epoch + 1) % self.save_interval == 0:
            self._save_data()
            
            print(f"\n{'='*80}")
            print(f"💾 Epoch {epoch}: 数据已保存")
            print(f"   AP50: {ap50:.4f}" if ap50 else "   AP50: N/A")
            print(f"   平均梯度Norm: {avg_grad_norm:.6f}")
            print(f"   各类别梯度Norm:")
            for cls_name, grad_norm in zip(model.get_cls_names(), avg_per_class_grads):
                print(f"      {cls_name:20s}: {grad_norm:.6f}")
            print(f"{'='*80}\n")
        
        # 清空
        self.current_epoch_grad_norms = []
        self.current_epoch_per_class_grads = []
        self._last_ap50 = None
    
    def after_train(self, runner):
        """训练结束"""
        self._save_data()
        
        if self.tb_writer is not None:
            self.tb_writer.close()
        
        print(f"\n✅ 训练完成! 数据已保存到: {self.data_file}")
        print(f"   - Embedding快照: {len(self.data['embeddings']['epochs'])} 个")
        print(f"   - 当前类别: {self.data['metadata']['cur_class_names']}")
        print(f"   - 之前类别: {self.data['metadata']['prev_class_names']}")
        
        # 打印AP50趋势
        valid_ap50 = [x for x in self.data['metrics']['ap50'] if x is not None]
        if valid_ap50:
            print(f"   - AP50: 初始={valid_ap50[0]:.2f}, 最终={valid_ap50[-1]:.2f}")
    
    def _save_data(self):
        """保存pickle文件"""
        with open(self.data_file, 'wb') as f:
            pickle.dump(self.data, f)