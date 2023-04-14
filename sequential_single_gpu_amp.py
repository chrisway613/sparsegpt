import os
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy

from accelerate.logging import get_logger
from torch.cuda.amp import autocast, GradScaler

from sparsegpt import SparseGPT
from modelutils import find_layers


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True 


class Catcher(nn.Module):
    def __init__(self, module, cache, states):
        super().__init__()

        self.module = module
        self.cache = cache
        self.states = states

    def forward(self, x, **kwargs):
        self.states[self.cache['layer_i']] = x
        self.cache['layer_i'] += 1
        # self.states.data = x.data
        self.cache['attention_mask'] = kwargs['attention_mask']
        self.cache['alibi'] = kwargs['alibi']

        # 为了“截断”模型的前向过程，从而仅 forward 到该模块就中止
        raise ValueError


class BatchCatcher(Catcher):
    def __init__(self, module, cache, states):
        super().__init__(module, cache, states)

    def forward(self, x, **kwargs):
        self.states.data = x.data
        self.cache['attention_mask'] = kwargs['attention_mask']
        self.cache['alibi'] = kwargs['alibi']

        # 为了“截断”模型的前向过程，从而仅 forward 到该模块就中止
        raise ValueError
    

class SequentialForward:
    def __init__(self, model, accelerator, logger=None):
        self.model = model

        self.dev = accelerator.device
        self.accelerator = accelerator
        self.logger = logger or get_logger("Sequential")

    def train(self, batch, lr=0., prune=False, **kwargs):
        raise NotImplementedError

    @torch.no_grad()
    def eval(self, batch, **kwargs):
        raise NotImplementedError

    def forward(self, batch, lr=0., train=True, prune=False, **kwargs):
        return self.train(batch, lr=lr, prune=prune, **kwargs) if train else self.eval(batch, **kwargs)


class SparseBloomSequential(SequentialForward):
    def __init__(self, model, accelerator, logger=None, dense_dir=None, sparse_thresh=1e-10):
        super().__init__(model, accelerator, logger=logger)

        # Path to dense weights
        self.dense_dir = dense_dir
        self.sparse_mask = {}
        self.sparse_thresh = sparse_thresh
        
        # Only train BloomBlocks
        self.model.transformer.word_embeddings.requires_grad_(False)
        self.model.transformer.word_embeddings_layernorm.requires_grad_(False)
        self.model.transformer.ln_f.requires_grad_(False)
        self.model.lm_head.requires_grad_(False)

        self.scaler = GradScaler(growth_interval=50)
    
    def forward_batch(self, batch, lr=0., train=True, prune=False, **kwargs):
        return self.train_batch(batch, lr=lr, prune=prune, **kwargs) if train else self.eval_batch(batch, **kwargs)

    def train(self, batch, lr=0., prune=False, **prune_kwargs):
        self.model.train()

        use_cache = self.model.config.use_cache
        self.model.config.use_cache = False

        # All BloomBlocks
        layers = self.model.transformer.h

        ''' CPU -> GPU '''

        # Embedding layer & the 1st Layer Norm
        self.model.transformer.word_embeddings.to(self.dev)
        self.model.transformer.word_embeddings_layernorm.to(self.dev)
        # The 1st BloomBlock
        layers[0].to(self.dev)

        dtype = next(iter(self.model.parameters())).dtype
        hidden_size = self.model.config.hidden_size
        num_samples, seq_length = batch['input_ids'].shape[:2]

        # 记录每个样本在输入 BloomBlock 前的 hidden state
        in_hs = torch.zeros(
            (num_samples, seq_length, hidden_size),
            dtype=dtype, device='cpu'
        )

        # 记录当前 forward 到第几个 BloomBlock 以及对应的一些状态
        cache = {'layer_i': 0, 'attention_mask': None, 'alibi': None}
        # 以下会仅经过了：Embedding 层 -> LayerNorm -> the 1st BloomBlock
        # 从而将每个样本在输入第一个 BloomBlock 前的 hidden state 记录下来
        layers[0] = Catcher(layers[0], cache, in_hs)
        # BloomBlock 前的部分(Embedding 层)不需要梯度
        with torch.no_grad():
            for i in range(num_samples):
                sample = {k: v[i].unsqueeze(0).to(self.dev) for k, v in batch.items()}
                try:
                    self.model(**sample)
                except ValueError:
                    pass

        layers[0] = layers[0].module.to('cpu')
        self.model.transformer.word_embeddings.to('cpu')
        self.model.transformer.word_embeddings_layernorm.to('cpu')

        # 记录每个样本经过 BloomBlock 输出后的 hidden states
        alibi = cache['alibi']
        attention_mask = cache['attention_mask']

        if not os.path.exists(self.dense_dir):
            raise RuntimeError(f"Path to dense weights '{self.dense_dir}' not existed.")

        # Cosine similarity between hidden states of sparse & dense
        similarities = []
        # Loss yeild by each layer
        layer_losses = []

        if prune:
            sparsity = prune_kwargs.pop('sparsity')
            min_layer = prune_kwargs.pop('min_layer', 0)
            max_layer = prune_kwargs.pop('max_layer', 1000)
            prune_only = prune_kwargs.pop('prune_only', '')

        for i in range(len(layers)):
            layer = layers[i]

            # Load dense weights
            teacher_layer = deepcopy(layer)
            layer_state_dict = torch.load(os.path.join(self.dense_dir, f'layer_{i}'), map_location='cpu')
            teacher_layer.load_state_dict(layer_state_dict)
            teacher_layer.to(self.dev)
            teacher_layer.eval()

            # Output hidden states of teacher, this will be regarded as the input hidden states for next layer
            teacher_out_hs = torch.zeros_like(in_hs)
            with torch.no_grad():
                for j in range(num_samples):
                    # NOTE: [0] means extracting hidden states from the output tuple
                    teacher_out_hs[j] = teacher_layer(in_hs[j].unsqueeze(0).to(self.dev), attention_mask=attention_mask, alibi=alibi)[0]
            
            del teacher_layer
            torch.cuda.empty_cache()
            gc.collect()

            layer.to(self.dev)
            # Pruning
            if prune:
                # 返回一个字典，找出当前 BloomBlock 下的所有 Linear 层
                subset = find_layers(layer)

                gpts = {}
                for name in subset:
                    if not (min_layer <= i < max_layer and prune_only in name):
                        continue

                    gpts[name] = SparseGPT(subset[name])

                def add_batch(name):
                    def tmp(_, inp, out):
                        # inp 是 tuple，取第一个代表 input_ids
                        gpts[name].add_batch(inp[0].data, out.data)

                    return tmp
            
                handles = [subset[name].register_forward_hook(add_batch(name)) for name in gpts]
                # 其实这里的目的并非记录每个样本经过当前 BloomBlock 输出后的 hidden state
                # (真正的记录过程在后面)
                # 而是为了 SparseGPT() 做 add_batch()，让前面的注册的 hook 发挥作用
                for j in range(num_samples):
                    layer(in_hs[j].unsqueeze(0).to(self.dev), attention_mask=attention_mask, alibi=alibi)
                for h in handles:
                    h.remove()

                # 对当前 BloomBlock 中的每个 Linear 层做 pruning
                self.logger.info(f"Pruning layer{i}..")
                for name in gpts:
                    self.logger.info(f"Module: {name}")
                    gpts[name].fasterprune(sparsity, **prune_kwargs)
                    # Binary sparse mask (0|1)
                    self.sparse_mask[name] = (subset[name].weight.abs() > self.sparse_thresh).to(dtype=dtype, device='cpu')
                self.logger.info("Done!\n")
                del gpts
            # Align with teacher
            else:
                optimizer = torch.optim.AdamW(layer.parameters(), lr=lr)
                optimizer.zero_grad(set_to_none=True)

                out_hs = torch.zeros_like(in_hs, device=self.dev)
                for j in range(num_samples):
                    # NOTE: [0] means extracting hidden states from the output tuple
                    out_hs[j] = layer(in_hs[j].unsqueeze(0).to(self.dev), attention_mask=attention_mask, alibi=alibi)[0]
                var = out_hs.pow(2).mean(dim=-1, keepdim=True)

                teacher_out_hs = teacher_out_hs.to(device=self.dev)
                teacher_var = teacher_out_hs.pow(2).mean(dim=-1, keepdim=True)

                # 每一层在每个 batch 上的 loss
                # 在算 loss 前做 layernorm 方式的归一化
                loss = F.mse_loss(
                    out_hs / (var + 1e-8),
                    teacher_out_hs / (teacher_var + 1e-8)
                )
                loss.backward()
                layer_losses.append(loss.item())
                del loss, var, teacher_var

                with torch.no_grad():
                    # Cosine similarity between sparse & dense
                    sim = F.cosine_similarity(
                        out_hs.reshape(-1, hidden_size),
                        teacher_out_hs.reshape(-1, hidden_size)
                    ).mean()
                
                del out_hs
                # Reduce across all process
                sim = (self.accelerator.reduce(sim) / self.accelerator.num_processes).item()
                similarities.append(sim)

                # Update parameters
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                del optimizer

                # Apply mask
                if self.sparse_mask:
                    subset = find_layers(layer)
                    for name, module in subset.items():
                        mask = self.sparse_mask[name].to(module.weight.device)
                        module.weight.data = module.weight.data * mask

            layers[i] = layer.to('cpu')
            del layer

            # Cut out the relationship between layers
            in_hs = teacher_out_hs.detach().cpu()
            del teacher_out_hs

            torch.cuda.empty_cache()
            gc.collect()

        del in_hs
        torch.cuda.empty_cache()
        gc.collect()

        self.model.config.use_cache = use_cache

        return layer_losses, similarities
   
    def train_batch(self, batch, lr=0., prune=False, **prune_kwargs):
        self.model.train()

        use_cache = self.model.config.use_cache
        self.model.config.use_cache = False

        # All BloomBlocks
        layers = self.model.transformer.h

        ''' CPU -> GPU '''

        # Embedding layer & the 1st Layer Norm
        self.model.transformer.word_embeddings.to(self.dev)
        self.model.transformer.word_embeddings_layernorm.to(self.dev)
        # The 1st BloomBlock
        layers[0].to(self.dev)

        # dtype = next(iter(self.model.parameters())).dtype
        dtype = torch.float16
        hidden_size = self.model.config.hidden_size
        num_samples, seq_length = batch['input_ids'].shape[:2]

        # 记录每个样本在输入 BloomBlock 前的 hidden state
        in_hs = torch.zeros(
            (num_samples, seq_length, hidden_size),
            dtype=dtype, device=('cpu' if prune else self.dev)
        )

        # 记录当前 forward 到第几个 BloomBlock 以及对应的一些状态
        cache = {'layer_i': 0, 'attention_mask': None, 'alibi': None}
        # 以下会仅经过了：Embedding 层 -> LayerNorm -> the 1st BloomBlock
        # 从而将每个样本在输入第一个 BloomBlock 前的 hidden state 记录下来
        layers[0] = Catcher(layers[0], cache, in_hs) if prune else BatchCatcher(layers[0], cache, in_hs)
        # BloomBlock 前的部分(Embedding 层)不需要梯度
        with torch.no_grad():
            # ->fp16
            with autocast():
                if prune:
                    for i in range(num_samples):
                        sample = {k: v[i].unsqueeze(0).to(self.dev) for k, v in batch.items()}
                        try:
                            self.model(**sample)
                        except ValueError:
                            pass
                else:
                    try:
                        self.model(**batch)
                    except ValueError:
                        pass

        layers[0] = layers[0].module.to('cpu')
        self.model.transformer.word_embeddings.to('cpu')
        self.model.transformer.word_embeddings_layernorm.to('cpu')

        # 记录每个样本经过 BloomBlock 输出后的 hidden states
        alibi = cache['alibi']
        attention_mask = cache['attention_mask']

        if not os.path.exists(self.dense_dir):
            raise RuntimeError(f"Path to dense weights '{self.dense_dir}' not existed.")

        # Cosine similarity between hidden states of sparse & dense
        similarities = []
        # Loss yeild by each layer
        layer_losses = []

        if prune:
            sparsity = prune_kwargs.pop('sparsity')
            min_layer = prune_kwargs.pop('min_layer', 0)
            max_layer = prune_kwargs.pop('max_layer', 1000)
            prune_only = prune_kwargs.pop('prune_only', '')

        for i in range(len(layers)):
            layer = layers[i]

            # Load dense weights
            teacher_layer = deepcopy(layer)
            layer_state_dict = torch.load(os.path.join(self.dense_dir, f'layer_{i}'), map_location='cpu')
            teacher_layer.load_state_dict(layer_state_dict)
            teacher_layer.to(self.dev)
            teacher_layer.eval()

            with torch.no_grad():
                with autocast():
                    # Output hidden states of teacher, this will be regarded as the input hidden states for next layer
                    # NOTE: [0] means extracting hidden states from the output tuple
                    if prune:
                        teacher_out_hs = torch.zeros_like(in_hs)
                        for j in range(num_samples):
                            teacher_out_hs[j] = teacher_layer(in_hs[j].unsqueeze(0).to(self.dev), attention_mask=attention_mask, alibi=alibi)[0]
                    else:
                        teacher_out_hs = teacher_layer(in_hs, attention_mask=attention_mask, alibi=alibi)[0].cpu()

            del teacher_layer
            torch.cuda.empty_cache()
            gc.collect()

            layer.to(self.dev)
            # Pruning
            if prune:
                # 返回一个字典，找出当前 BloomBlock 下的所有 Linear 层
                subset = find_layers(layer)

                gpts = {}
                for name in subset:
                    if not (min_layer <= i < max_layer and prune_only in name):
                        continue

                    gpts[name] = SparseGPT(subset[name])

                def add_batch(name):
                    def tmp(_, inp, out):
                        # inp 是 tuple，取第一个代表 input_ids
                        gpts[name].add_batch(inp[0].data, out.data)

                    return tmp
            
                handles = [subset[name].register_forward_hook(add_batch(name)) for name in gpts]
                # 其实这里的目的并非记录每个样本经过当前 BloomBlock 输出后的 hidden state
                # (真正的记录过程在后面)
                # 而是为了 SparseGPT() 做 add_batch()，让前面的注册的 hook 发挥作用
                for j in range(num_samples):
                    layer(in_hs[j].unsqueeze(0).to(device=self.dev, dtype=torch.float32), attention_mask=attention_mask, alibi=alibi)
                for h in handles:
                    h.remove()

                # 对当前 BloomBlock 中的每个 Linear 层做 pruning
                self.logger.info(f"Pruning layer{i}..")
                for name in gpts:
                    self.logger.info(f"Module: {name}")
                    gpts[name].fasterprune(sparsity, **prune_kwargs)
                    # Binary sparse mask (0|1)
                    self.sparse_mask[name] = (subset[name].weight.abs() > self.sparse_thresh).to(
                        dtype=subset[name].weight.dtype,
                        device='cpu'
                    )
                self.logger.info("Done!\n")
                del gpts
            # Align with teacher
            else:
                optimizer = torch.optim.AdamW(layer.parameters(), lr=lr)
                optimizer.zero_grad(set_to_none=True)

                with autocast():
                    # NOTE: [0] means extracting hidden states from the output tuple
                    out_hs = layer(in_hs, attention_mask=attention_mask, alibi=alibi)[0]
                    in_hs = in_hs.to('cpu')
                    var = out_hs.pow(2).mean(dim=-1, keepdim=True)

                    teacher_out_hs = teacher_out_hs.to(self.dev)
                    teacher_var = teacher_out_hs.pow(2).mean(dim=-1, keepdim=True)

                    # 每一层在每个 batch 上的 loss
                    # 在算 loss 前做 layernorm 方式的归一化
                    loss = F.mse_loss(
                        out_hs / (var + 1e-6),
                        teacher_out_hs / (teacher_var + 1e-6)
                    )

                # Scale loss to fp16 range.
                # Calls backward() on scaled loss to create scaled gradients.
                self.scaler.scale(loss).backward()
                layer_losses.append(loss.item())
                del loss, var, teacher_var

                with torch.no_grad():
                    # Cosine similarity between sparse & dense
                    sim = F.cosine_similarity(
                        out_hs.reshape(-1, hidden_size).to(torch.float32),
                        teacher_out_hs.reshape(-1, hidden_size).to(torch.float32)
                    ).mean()
                
                del out_hs
                # Reduce across all process
                sim = (self.accelerator.reduce(sim) / self.accelerator.num_processes).item()
                similarities.append(sim)

                # scaler.step() first unscales gradients of the optimizer's params.
                # If gradients don't contain infs/NaNs, optimizer.step() is then called,
                # otherwise, optimizer.step() is skipped.
                self.scaler.step(optimizer)
                # Updates the scale for next iteration.
                self.scaler.update()

                # Zero gradients to release memory
                optimizer.zero_grad(set_to_none=True)
                del optimizer

                # Apply mask
                if self.sparse_mask:
                    subset = find_layers(layer)
                    for name, module in subset.items():
                        mask = self.sparse_mask[name].to(device=module.weight.device, dtype=module.weight.dtype)
                        module.weight.data = module.weight.data * mask

            layers[i] = layer.to('cpu')
            del layer

            # Cut out the relationship between layers
            in_hs = teacher_out_hs.detach().to('cpu' if prune else self.dev)
            del teacher_out_hs

            torch.cuda.empty_cache()
            gc.collect()

        del in_hs
        torch.cuda.empty_cache()
        gc.collect()

        self.model.config.use_cache = use_cache

        return layer_losses, similarities

    @torch.no_grad()
    def eval(self, batch, hard_sparse_weight=False, sparse_ratio=0.):
        self.model.eval()

        use_cache = self.model.config.use_cache
        self.model.config.use_cache = False

        # All BloomBlocks
        layers = self.model.transformer.h

        ''' CPU -> GPU '''

        # Embedding layer & the 1st Layer Norm
        self.model.transformer.word_embeddings.to(self.dev)
        self.model.transformer.word_embeddings_layernorm.to(self.dev)
        # The 1st BloomBlock
        layers[0].to(self.dev)

        dtype = next(iter(self.model.parameters())).dtype
        hidden_size = self.model.config.hidden_size
        num_samples, seq_length = batch['input_ids'].shape[:2]

        # 记录每个样本在输入 BloomBlock 前的 hidden state
        in_hs = torch.zeros(
            (num_samples, seq_length, hidden_size),
            dtype=dtype, device=self.dev
        )

        # 记录当前 forward 到第几个 BloomBlock 以及对应的一些状态
        cache = {'layer_i': 0, 'attention_mask': None, 'alibi': None}
        # 以下会仅经过了：Embedding 层 -> LayerNorm -> the 1st BloomBlock
        # 从而将每个样本在输入第一个 BloomBlock 前的 hidden state 记录下来
        layers[0] = Catcher(layers[0], cache, in_hs)

        for i in range(num_samples):
            sample = {k: v[i].unsqueeze(0) for k, v in batch.items()}
            try:
                self.model(**sample)
            except ValueError:
                pass

        layers[0] = layers[0].module.to('cpu')
        self.model.transformer.word_embeddings.to('cpu')
        self.model.transformer.word_embeddings_layernorm.to('cpu')

        alibi = cache['alibi']
        attention_mask = cache['attention_mask']

        for i in range(len(layers)):
            layer = layers[i].to(self.dev)
            # 将稀疏的部分置0
            if hard_sparse_weight:
                subset = find_layers(layer)
                for name in subset:
                    W = subset[name].weight.data
                    # 例如稀疏率是75%，那么先有小到大排序，然后将前 75% 的参数值置0
                    thresh = torch.sort(torch.abs(W.flatten()))[0][int(W.numel() * sparse_ratio)]
                    W.data[torch.abs(W.data) <= thresh] = 0

            for j in range(num_samples):
                # 每个 BloomBlock 的输出是元组，取第一个代表 hidden states
                in_hs[j] = layer(in_hs[j].unsqueeze(0), attention_mask=attention_mask, alibi=alibi)[0]

            layers[i] = layer.to('cpu')
            del layer

            torch.cuda.empty_cache()
            gc.collect()
        
        self.model.transformer.ln_f.to(self.dev)
        self.model.lm_head.to(self.dev)

        loss_fct = nn.CrossEntropyLoss()

        loss = 0.
        for k in range(num_samples):
            # Cast to float32 for better performance
            logits = self.model.lm_head(self.model.transformer.ln_f(in_hs[k].unsqueeze(0))).to(torch.float32)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch['labels'][k][..., 1:].contiguous().to(shift_logits.device)

            loss_per_sample = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss + loss_per_sample / num_samples
        del logits, shift_logits, shift_labels, loss_per_sample, in_hs

        self.model.transformer.ln_f.to('cpu')
        self.model.lm_head.to('cpu')

        self.model.config.use_cache = use_cache

        return loss
    
    @torch.no_grad()
    def eval_batch(self, batch, hard_sparse_weight=False, sparse_ratio=0.):
        self.model.eval()

        use_cache = self.model.config.use_cache
        self.model.config.use_cache = False

        # All BloomBlocks
        layers = self.model.transformer.h

        ''' CPU -> GPU '''

        # Embedding layer & the 1st Layer Norm
        self.model.transformer.word_embeddings.to(self.dev)
        self.model.transformer.word_embeddings_layernorm.to(self.dev)
        # The 1st BloomBlock
        layers[0].to(self.dev)

        # dtype = next(iter(self.model.parameters())).dtype
        dtype = torch.float16
        hidden_size = self.model.config.hidden_size
        num_samples, seq_length = batch['input_ids'].shape[:2]

        # 记录每个样本在输入 BloomBlock 前的 hidden state
        in_hs = torch.zeros(
            (num_samples, seq_length, hidden_size),
            dtype=dtype, device=self.dev
        )

        # 记录当前 forward 到第几个 BloomBlock 以及对应的一些状态
        cache = {'layer_i': 0, 'attention_mask': None, 'alibi': None}
        # 以下会仅经过了：Embedding 层 -> LayerNorm -> the 1st BloomBlock
        # 从而将每个样本在输入第一个 BloomBlock 前的 hidden state 记录下来
        layers[0] = BatchCatcher(layers[0], cache, in_hs)
        with autocast():
            try:
                self.model(**batch)
            except ValueError:
                pass

        layers[0] = layers[0].module.to('cpu')
        self.model.transformer.word_embeddings.to('cpu')
        self.model.transformer.word_embeddings_layernorm.to('cpu')

        alibi = cache['alibi']
        attention_mask = cache['attention_mask']

        for i in range(len(layers)):
            layer = layers[i].to(self.dev)
            # 将稀疏的部分置0
            if hard_sparse_weight:
                subset = find_layers(layer)
                for name in subset:
                    W = subset[name].weight.data
                    # 例如稀疏率是75%，那么先有小到大排序，然后将前 75% 的参数值置0
                    thresh = torch.sort(torch.abs(W.flatten()))[0][int(W.numel() * sparse_ratio)]
                    W.data[torch.abs(W.data) <= thresh] = 0

            with autocast():
                # 每个 BloomBlock 的输出是元组，取第一个代表 hidden states
                in_hs = layer(in_hs, attention_mask=attention_mask, alibi=alibi)[0]

            layers[i] = layer.to('cpu')
            del layer

            torch.cuda.empty_cache()
            gc.collect()
        
        self.model.transformer.ln_f.to(self.dev)
        self.model.lm_head.to(self.dev)

        with autocast():
            logits = self.model.lm_head(self.model.transformer.ln_f(in_hs))

        # Cast to float32 for better performance
        shift_logits = logits.to(torch.float32)[:, :-1, :].contiguous()
        shift_labels = batch['labels'][..., 1:].contiguous().to(self.dev)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        self.model.transformer.ln_f.to('cpu')
        self.model.lm_head.to('cpu')

        del in_hs, logits, shift_logits, shift_labels
        torch.cuda.empty_cache()
        gc.collect()

        self.model.config.use_cache = use_cache

        return loss
