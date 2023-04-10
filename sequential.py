import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy

from accelerate.logging import get_logger

from sparsegpt import SparseGPT
from modelutils import find_layers


class Catcher(nn.Module):
    def __init__(self, module, cache, states):
        super().__init__()

        self.module = module
        self.cache = cache
        self.states = states

    def forward(self, x, **kwargs):
        self.states[self.cache['layer_i']] = x

        self.cache['layer_i'] += 1
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

        # Output hidden states of each BloomBlock
        self.teacher_out_hs = [None] * len(self.model.transformer.h)

    def train(self, batch, lr=0., prune=False, **kwargs):
        raise NotImplementedError

    @torch.no_grad()
    def eval(self, batch, **kwargs):
        raise NotImplementedError

    def forward(self, batch, lr=0., train=True, prune=False, **kwargs):
        self.train(batch, lr=lr, prune=prune, **kwargs) if train else self.eval(batch, **kwargs)


class BloomSequential(SequentialForward):
    def __init__(self, model, accelerator, logger=None):
        super().__init__(model, accelerator, logger=logger)
        
        # Only train BloomBlocks
        self.model.transformer.word_embeddings.requires_grad_(False)
        self.model.transformer.word_embeddings_layernorm.requires_grad_(False)
        self.model.transformer.ln_f.requires_grad_(False)
        self.model.lm_head.requires_grad_(False)

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
        layers[0] = layers[0].to(self.dev)

        dtype = next(iter(self.model.parameters())).dtype
        hidden_size = self.model.config.hidden_size
        # seq_length = next(iter(dataloader))['input_ids'].size(1)
        # num_samples = dataloader.batch_size * len(dataloader)
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
        # BloomBlock 前的部分(Embedding 层)不需要梯度
        with torch.no_grad():
            for i in range(num_samples):
                sample = {k: v[i].unsqueeze(0) for k, v in batch.items()}
                try:
                    self.model(**sample)
                except ValueError:
                    pass

        layers[0] = layers[0].module.cpu()
        self.model.transformer.word_embeddings.cpu()
        self.model.transformer.word_embeddings_layernorm.cpu()
        
        torch.cuda.empty_cache()

        # 记录每个样本经过 BloomBlock 输出后的 hidden states
        # out_hs = torch.zeros_like(in_hs)
        alibi = cache['alibi']
        attention_mask = cache['attention_mask']

        for i in range(len(layers)):
            layer = layers[i]
            teacher_layer = deepcopy(layer)

            layer_state_dict = torch.load(f'layer_{i}', map_location='cpu')
            teacher_layer.load_state_dict(layer_state_dict)
            teacher_layer.to(self.dev)

            teacher_out_hs = teacher_layer(in_hs, attention_mask=attention_mask, alibi=alibi).cpu()
            teacher_layer.to('cpu')
            torch.cuda.empty_cache()

            # Collect teacher's output hidden states if not existed
            # if self.teacher_out_hs[i] is None:
            #     self.teacher_out_hs[i] = torch.zeros_like(in_hs, device='cpu')
            #     # 记录每个样本经过当前 BloomBlock 输出后的 hidden state
            #     # for j in range(num_samples):
            #     #     self.teacher_out_hs[j] = layer(
            #     #         in_hs[j].unsqueeze(0),
            #     #         attention_mask=attention_mask, alibi=alibi
            #     #     )[0].cpu()
            #     self.teacher_out_hs[i] = layer(in_hs, attention_mask=attention_mask, alibi=alibi).cpu()

            # Pruning
            if prune:                
                # 返回一个字典，找出当前 BloomBlock 下的所有 Linear 层
                subset = find_layers(layer)

                min_layer = prune_kwargs.pop('min_layer', 0)
                max_layer = prune_kwargs.pop('max_layer', 1000)
                prune_only = prune_kwargs.pop('prune_only', '')

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
                # for j in range(num_samples):
                #     layer(in_hs[j].unsqueeze(0), attention_mask=attention_mask, alibi=alibi)
                layer(in_hs, attention_mask=attention_mask, alibi=alibi)
                for h in handles:
                    h.remove()

                # 对当前 BloomBlock 中的每个 Linear 层做 pruning
                self.logger.info(f"Pruning layer{i}..")
                for name in gpts:
                    self.logger.info(f"Module: {name}")
                    sparsity = prune_kwargs.pop('sparsity')
                    gpts[name].fasterprune(sparsity, **prune_kwargs)
                self.logger.info("Done!\n")
                del gpts
            # Align with teacher
            else:
                optimizer = torch.optim.AdamW(layer.parameters(), lr=lr)

                # 记录每个样本经过当前 BloomBlock 输出后的 hidden state
                out_hs = layer(in_hs, attention_mask=attention_mask, alibi=alibi)
                loss = F.mse_loss(out_hs, self.teacher_out_hs[i].to(out_hs.device))
                loss.backward()

                # Update parameters
                optimizer.step()
                optimizer.zero_grad()

                del out_hs
                del optimizer

            layers[i] = layer.cpu()
            del layer
            torch.cuda.empty_cache()

            in_hs = self.teacher_out_hs[i].to(in_hs.device)

        del in_hs
        torch.cuda.empty_cache()

        self.model.config.use_cache = use_cache

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
        layers[0] = layers[0].to(self.dev)

        dtype = next(iter(self.model.parameters())).dtype
        hidden_size = self.model.config.hidden_size
        # seq_length = next(iter(dataloader))['input_ids'].size(1)
        # num_samples = dataloader.batch_size * len(dataloader)
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

        layers[0] = layers[0].module.cpu()
        self.model.transformer.word_embeddings.cpu()
        self.model.transformer.word_embeddings_layernorm.cpu()
        
        torch.cuda.empty_cache()

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

            in_hs = layer(in_hs, attention_mask=attention_mask, alibi=alibi)
            layers[i] = layer.cpu()

            del layer
            torch.cuda.empty_cache()
        
        self.model.transformer.ln_f.to(self.dev)
        self.model.lm_head.to(self.dev)

        logits = self.model.lm_head(self.model.transformer.ln_f(in_hs))
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch['labels'][..., 1:].contiguous().to(shift_logits.device)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        del logits, shift_logits, shift_labels
        self.model.transformer.ln_f.to('cpu')
        self.model.lm_head.to('cpu')

        torch.cuda.empty_cache()
        self.model.config.use_cache = use_cache

        return loss
