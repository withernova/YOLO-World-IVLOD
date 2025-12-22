import torch
from contextlib import contextmanager

class CLIPEmbeddingHook:
    def __init__(self,clip_model ,n_ctx=16, start_token=1):
        self.clip_model = clip_model
        self.start_token = start_token
        self.n_ctx = n_ctx
        self.cur_embedding = None
        self.max_len = 77
        self.hooks= []
        
    def set_embedding(self, cur_embedding):
        self.cur_embedding = cur_embedding
        
    def _hook_fn(self, module, inputs, output):
        # output: (Batch, Seq_Len, Dim)
        if self.cur_embedding is None:
            return output
        
        learned_embedding = self.cur_embedding.to(dtype=output.dtype, device=output.device)
        
        if learned_embedding.dim() == 2:
            learned_embedding = learned_embedding.unsqueeze(0) 
            
        modified_output = output.clone()
        modified_output[:, 1 : 1 + self.n_ctx, :] = learned_embedding
        #modified_output[:, 1:2, :] = learned_embedding
        
        return modified_output
    
    def _hook_pre_forward(self, module, args, kwargs):
        if 'input_ids' in kwargs:
            input_ids = kwargs['input_ids']
        elif len(args) > 0:
            input_ids = args[0]
        else:
            return args, kwargs

        if 'attention_mask' in kwargs:
            mask = kwargs['attention_mask']
        elif len(args) > 1:
            mask = args[1]
        else:
            mask = torch.ones_like(input_ids)

        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        dummy_ids = torch.zeros((batch_size, self.n_ctx), dtype=input_ids.dtype, device=device)
        dummy_mask = torch.ones((batch_size, self.n_ctx), dtype=mask.dtype, device=device)
        
        prefix_ids = input_ids[:, :1]
        suffix_ids = input_ids[:, 1:]
        new_input_ids = torch.cat([prefix_ids, dummy_ids, suffix_ids], dim=1)
        
        prefix_mask = mask[:, :1]
        suffix_mask = mask[:, 1:]
        new_mask = torch.cat([prefix_mask, dummy_mask, suffix_mask], dim=1)
        
        if new_input_ids.shape[1] > self.max_len:
            new_input_ids = new_input_ids[:, :self.max_len]
            new_mask = new_mask[:, :self.max_len]
            
        if 'input_ids' in kwargs:
            kwargs['input_ids'] = new_input_ids
        else:
            args_list = list(args)
            args_list[0] = new_input_ids
            args = tuple(args_list)
            
        if 'attention_mask' in kwargs:
            kwargs['attention_mask'] = new_mask
        elif len(args) > 1:
            args_list = list(args)
            args_list[1] = new_mask
            args = tuple(args_list)
            
        return args, kwargs

    def __enter__(self):
        target_module = self.clip_model.model.text_model.embeddings.token_embedding    
        self.hooks.append(target_module.register_forward_hook(self._hook_fn))
        
        self.hooks.append(
            self.clip_model.model.text_model.register_forward_pre_hook(
                self._hook_pre_forward, with_kwargs=True
            )
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for h in self.hooks:
            h.remove()
        self.hooks = []