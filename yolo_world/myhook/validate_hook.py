from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.registry import HOOKS

@HOOKS.register_module()
class IterValHook(Hook):
    def __init__(self, interval=20, startiter=10):
        self.startiter = startiter
        self.interval = interval

    def after_train_iter(self, runner: Runner, batch_idx: int, data_batch=None, outputs=None):
        current_iter = runner.iter + 1
        
        if current_iter > self.startiter and current_iter % self.interval == 0:
            runner.logger.info(f'\n🔄 Triggering validation at iter {current_iter}...')
            
            runner.val_loop.run()
            
            runner.model.train() 