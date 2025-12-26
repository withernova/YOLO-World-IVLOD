import torch
from contextlib import contextmanager

class CLIPEmbeddingHook:
    def __init__(self, clip_model, n_ctx=16, start_token=1):
        self.clip_model = clip_model
        self.start_token = start_token
        self.n_ctx = n_ctx
        self.batch_embeddings = None # 存储处理后的 tensor
        self.embedding_mask = None   # 存储对应的 mask
        self.insert_len = 0          # 当前 batch 最大的插入长度
        self.max_len = 77
        self.hooks = []
        
    def set_embedding(self, embedding_list):
        if embedding_list is None or len(embedding_list) == 0:
            self.batch_embeddings = None
            return

        valid_tensors = [e for e in embedding_list if e is not None]
        
        if not valid_tensors:
            self.batch_embeddings = None
            self.insert_len = 0
            return

        ref_tensor = valid_tensors[0]
        device = ref_tensor.device
        dtype = ref_tensor.dtype

        lengths = [e.shape[0] if e is not None else 0 for e in embedding_list]
        self.insert_len = max(lengths) 

        if self.insert_len == 0:
            self.batch_embeddings = None
            return

        batch_size = len(embedding_list)
        
        padded_emb = torch.zeros((batch_size, self.insert_len, ref_tensor.shape[1]), 
                                 device=device, dtype=dtype)
        emb_mask = torch.zeros((batch_size, self.insert_len), device=device, dtype=torch.long)

        for i, emb in enumerate(embedding_list):
            if emb is None:
                continue
            
            L = emb.shape[0]
            if emb.device != device:
                emb = emb.to(device)
                
            padded_emb[i, :L, :] = emb # 填入真实 embedding
            emb_mask[i, :L] = 1        # 仅将有数据的部分标记为 1

        self.batch_embeddings = padded_emb
        self.embedding_mask = emb_mask
        
    def _hook_fn(self, module, inputs, output):
        """
        替换 Token Embedding 的输出。
        """
        if self.batch_embeddings is None:
            return output
        
        learned_embedding = self.batch_embeddings.to(dtype=output.dtype, device=output.device)
        
        modified_output = output.clone()
        modified_output[:, self.start_token : self.start_token + self.insert_len, :] = learned_embedding
        
        return modified_output
    
    def _hook_pre_forward(self, module, args, kwargs):
        if self.batch_embeddings is None:
            return args, kwargs

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
        
        dummy_ids = torch.zeros((batch_size, self.insert_len), dtype=input_ids.dtype, device=input_ids.device)
        
        prefix_ids = input_ids[:, :self.start_token]
        suffix_ids = input_ids[:, self.start_token:]
        new_input_ids = torch.cat([prefix_ids, dummy_ids, suffix_ids], dim=1)

        # 拼接 attention_mask
        # 使用我们在 set_embedding 计算好的 mask
        # 对于 None 的行，中间这段全是 0，意味着模型会忽略这段插入的 dummy tokens
        current_emb_mask = self.embedding_mask.to(mask.device) # (B, Insert_Len)
        
        prefix_mask = mask[:, :self.start_token]
        suffix_mask = mask[:, self.start_token:]
        
        new_mask = torch.cat([prefix_mask, current_emb_mask, suffix_mask], dim=1)

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