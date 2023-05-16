import os
import gc
import json
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from copy import deepcopy

from transformers import BloomForCausalLM

from datautils import get_loaders
from modelutils import find_layers
from sparsegpt import SparseGPT, ABCSolver


try:
    import wandb
    has_wandb = True
except:
    has_wandb = False    


def get_bloom(model, seq_len=256, model_dtype=torch.bfloat16):
    
    def skip(*args, **kwargs):
        pass

    # TODO: what do these attempt to do?
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    
    model = BloomForCausalLM.from_pretrained(
        model, 
        cache_dir=args.model_cache_dir, 
        torch_dtype=model_dtype
    )
    model.seqlen = seq_len
    
    return model


@torch.no_grad()
def bloom_sequential(model, dataloader, dev='cpu', abc_solver=False):
    print('Starting..\n')

    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.transformer.h

    model.transformer.word_embeddings = model.transformer.word_embeddings.to(DEV)
    model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.to(DEV)
    # The 1st BloomBlock
    layers[0] = layers[0].to(DEV)

    dtype = next(iter(model.parameters())).dtype
    # 记录每个样本在输入 BloomBlock 前的 hidden state
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size),
        dtype=dtype, device=dev
    )
    # 记录当前 forward 到第几个 BloomBlock 以及对应的一些状态
    cache = {'i': 0, 'attention_mask': None, 'alibi': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1

            cache['attention_mask'] = kwargs['attention_mask']
            cache['alibi'] = kwargs['alibi']

            # 为了“截断”模型的前向过程，从而仅 forward 到该模块就中止
            raise ValueError
        
    # 以下会仅经过了：Embedding 层 -> LayerNorm -> the 1st BloomBlock
    # 从而将每个样本在输入第一个 BloomBlock 前的 hidden state 记录下来
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            # batch 是 (input_id, target)，它们的 shape 均是 (1, seq_length)
            model(batch[0].to(DEV))
        except ValueError:
            pass

    layers[0] = layers[0].module.cpu()
    model.transformer.word_embeddings = model.transformer.word_embeddings.cpu()
    model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.cpu()
    
    torch.cuda.empty_cache()

    attention_mask = cache.pop('attention_mask')
    alibi = cache.pop('alibi')
    del cache

    # 记录每个样本经过 BloomBlock 输出后的 hidden states
    outs = torch.zeros_like(inps)

    with open(os.path.join(args.save, 'pytorch_model.bin.index.json')) as f:
        weight_index = deepcopy(json.load(f))

    print('Ready!\n')

    for i in range(len(layers)):
        layer = layers[i].to(DEV)
        # 返回一个字典，找出当前 BloomBlock 下的所有 Linear 层
        # subset = find_layers(layer)
        # TODO
        subset = find_layers(layer, name=f'h.{i}')

        gpts = {}
        for name in subset:
            if (not (args.minlayer <= i < args.maxlayer and args.prune_only in name)) == (not args.invert):
                continue
            gpts[name] = ABCSolver(subset[name]) if abc_solver else SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                # inp 是 tuple，取第一个代表 input_ids
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp
        
        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        # 其实这里的目的并非记录每个样本经过当前 BloomBlock 输出后的 hidden state
        # (真正的记录过程在后面)
        # 而是为了 SparseGPT() 做 add_batch()，让前面的注册的 hook 发挥作用
        for j in range(args.nsamples):
            # Outptus from dense
            outs[j] = layer(inps[j].unsqueeze(0).to(DEV), attention_mask=attention_mask, alibi=alibi)[0]
            # inps[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, alibi=alibi)[0]
            # layer(inps[j].unsqueeze(0), attention_mask=attention_mask, alibi=alibi)

        for h in handles:
            h.remove()

        # 对当前 BloomBlock 中的每个 Linear 层做 pruning
        print('Pruning..')
        for name in gpts:
            print(f"layer: {i}\tname: {name}")

            if abc_solver:
                gpts[name].prune_structured(
                    args.sparsity,
                    percdamp=args.percdamp
                )
            else:
                gpts[name].fasterprune(
                    args.sparsity,
                    prunen=args.prunen,
                    prunem=args.prunem,
                    percdamp=args.percdamp
                )

        print("Done!\n")

        # Pruning 后，记录每个样本经过当前 BloomBlock 输出后的 hidden state
        for j in range(args.nsamples):
            # Outputs from sparse
            inps[j] = layer(inps[j].unsqueeze(0).to(DEV), attention_mask=attention_mask, alibi=alibi)[0]

        layers[i] = layer.cpu()

        del gpts, layer
        gc.collect()
        torch.cuda.empty_cache()

        # Cosine similarity between sparse & dense.
        sim = 0.
        for sp, ds in zip(inps, outs):
            per_sim = F.cosine_similarity(sp.to(device=DEV, dtype=torch.float32), ds.to(device=DEV, dtype=torch.float32)).mean().item()
            sim += per_sim / args.nsamples
        print(f"[Layer{i}] Cosine similarity between sparse & dense: {sim}\n")

        # Save the state dict of current layer
        dst = os.path.join(args.save, f"pytorch_model_layer{i}.bin")
        sd = layers[i].state_dict(prefix=f"h.{i}.")
        torch.save(sd, dst)
        print(f"Sparse state dict of layer{i} has been saved to {dst}\n")

        sd_map = {}.fromkeys(sd.keys(), os.path.basename(dst))
        weight_index['weight_map'].update(sd_map)

        del sd, sd_map
        gc.collect()

        # Inputs of the next layer comes from the outputs of current dense layer.
        inps = outs.clone()

    del inps, outs
    gc.collect()
    torch.cuda.empty_cache()

    model.config.use_cache = use_cache

    with open(os.path.join(args.save, 'pytorch_model.bin.index.json'), 'w') as f:
        json.dump(weight_index, f)
    print(f"Weight index already dump to {os.path.join(args.save, 'pytorch_model.bin.index.json')}")


@torch.no_grad()
def bloom_eval(model, testenc, dataset: str, dev='cpu', log_wandb: bool = False):
    print('Evaluation..\n')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.transformer.h

    model.transformer.word_embeddings = model.transformer.word_embeddings.to(DEV)
    model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.to(DEV)
    layers[0] = layers[0].to(DEV)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size),
        dtype=dtype, device=dev
    )

    cache = {'i': 0, 'attention_mask': None, 'alibi': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['alibi'] = kwargs['alibi']

            raise ValueError
        
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(DEV)
        try:
            model(batch)
        except ValueError:
            pass

    layers[0] = layers[0].module.cpu()
    model.transformer.word_embeddings = model.transformer.word_embeddings.cpu()
    model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.cpu()

    torch.cuda.empty_cache()

    attention_mask = cache.pop('attention_mask')
    alibi = cache.pop('alibi')
    del cache

    for i in tqdm(range(len(layers)), desc="Sequential Forward"):
        layer = layers[i].to(DEV)

        # 将稀疏的部分置0
        if args.gmp:
            subset = find_layers(layer)
            for name in subset:
                W = subset[name].weight.data
                # 例如稀疏率是75%，那么先有小到大排序，然后将前 75% 的参数值置0
                thresh = torch.sort(torch.abs(W.flatten()))[0][int(W.numel() * args.sparsity)]
                W.data[torch.abs(W.data) <= thresh] = 0

        for j in range(nsamples):
            inps[j] = layer(inps[j].unsqueeze(0).to(DEV), attention_mask=attention_mask, alibi=alibi)[0]

        layers[i] = layer.cpu()
        del layer
        
        gc.collect()
        torch.cuda.empty_cache()

    model.transformer.ln_f = model.transformer.ln_f.to(DEV)
    model.lm_head = model.lm_head.to(DEV)

    loss_fct = nn.CrossEntropyLoss()

    nlls = []
    for i in tqdm(range(nsamples), desc="Per sample evaluation"):
        hidden_states = inps[i].unsqueeze(0).to(DEV)
        # hidden_states = model.transformer.ln_f(hidden_states)
        # lm_logits = model.lm_head(hidden_states)
        lm_logits = model.lm_head(model.transformer.ln_f(hidden_states))

        shift_logits = lm_logits[:, :-1, :].contiguous()
        # shift_labels = testenc[
        #     :, (i * model.seqlen):((i + 1) * model.seqlen)
        # ][:, 1:]
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:].to(DEV)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)

        del loss
        gc.collect()
    
    model.transformer.ln_f = model.transformer.ln_f.cpu()
    model.lm_head = model.lm_head.cpu()

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"Perplexity over {nsamples} samples: {ppl.item():3f}\n")

    if log_wandb:
         wandb.log({f'{dataset}/perplexity': ppl.item()})

    model.config.use_cache = use_cache


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='BLOOM model to load; pass `bigscience/bloom-X`.'
    )
    parser.add_argument(
        'dataset', type=str, choices=['wikipedia', 'wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="model identifier from huggingface.co/models, used for initial tokenizer."
    )
    parser.add_argument(
        "--model_cache_dir",
        type=str,
        default=None,
        help="Directory of model weights to be cached in."
    )
    parser.add_argument(
        "--data_cache_dir",
        type=str,
        default=None,
        help="Directory of the dataset to be cached.",
    )
    parser.add_argument(
        '--seed',
        type=int, default=42, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=256,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        "--abc_solver",
        action="store_true",
        help="Whether to use ABC Solver(default is SparseGPT)."
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--sparsity', type=float, default=0,
        help='Target sparsity'
    )
    parser.add_argument(
        '--prunen', type=int, default=0,
        help='N for N:M pruning.'
    )
    parser.add_argument(
        '--prunem', type=int, default=0,
        help='M for N:M pruning.'
    )
    parser.add_argument(
        '--gmp', action='store_true',
        help='Whether to run the GMP baseline.'
    )
    parser.add_argument(
        '--minlayer', type=int, default=-1,
        help='Prune all layers with id >= this.'
    )
    parser.add_argument(
        '--maxlayer', type=int, default=1000,
        help='Prune all layers with id < this.'
    )
    parser.add_argument(
        '--prune_only', type=str, default='',
        help='Prune only layers that contain this text.'
    )
    parser.add_argument(
       '--invert', action='store_true',
       help='Invert subset.'
    )
    parser.add_argument(
       '--save', type=str, default='',
       help='Path to saved model.'
    )
    parser.add_argument(
       '--log_wandb', action='store_true',
       help='Whether to log to wandb.'
    )

    parser.add_argument(
        '--cuda_id',
        type=int,
        default=0,
        help='Index of the cuda device to be used.'
    )
    parser.add_argument(
        '--eval_dense',
        action='store_true',
        help="Whether to do evaluation for dense."
    )
    parser.add_argument(
        "--eval_sparse",
        action="store_true",
        help="Whether to do evaluation after pruning."
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Whether to turn on debug mode. If true, only 1000 samples will be selected for training data.'
    )

    args = parser.parse_args()
    # Init W&B logging
    if args.log_wandb:
        assert has_wandb, "wandb not installed try `pip install wandb`"
        wandb.init(config=args)

    return args


if __name__ == '__main__':
    args = parse_args()
    print(f"Target sparsity: {args.sparsity}\n")

    DEV = torch.device(f'cuda:{args.cuda_id}')

    model = get_bloom(args.model)
    model.eval()

    # dataloader, testloader = get_loaders(
    #     args.dataset, nsamples=args.nsamples,
    #     model=args.model_name, cache_dir=args.model_cache_dir,
    #     seqlen=model.seqlen, seed=args.seed, debug=args.debug
    # )
    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, data_cache_dir=args.data_cache_dir,
        model=args.model_name, tokenizer_cache_dir=args.model_cache_dir,
        seqlen=model.seqlen, seed=args.seed, debug=args.debug,
        test=True  # load testset
    )
    # NOTE: this is for use all data
    args.nsamples = len(dataloader)

    if args.eval_dense:
        print("[Eval for dense]")
        bloom_eval(model, testloader, args.dataset, log_wandb=args.log_wandb)

    if (args.sparsity or args.prunen) and not args.gmp:
        bloom_sequential(model, dataloader, abc_solver=args.abc_solver)
        # Check sparsity
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and 'head' not in name:
                print(f"Module: {name}\t Sparsity: {torch.mean((module.weight == 0).float())}")

    if args.eval_sparse:
        # for dataset in ['wikitext2', 'ptb', 'c4']:
            # _, testloader = get_loaders(
            #     dataset, seed=args.seed,
            #     model=args.model, seqlen=model.seqlen
            # )
            # print(f"Dataset: {dataset}")
            # bloom_eval(model, testloader, DEV, dataset, args.log_wandb)
        
        # TODO: remove future, for debug currentl
        bloom_eval(model, testloader, args.dataset, log_wandb=args.log_wandb)
        # bloom_eval(model, testloader, DEV, args.dataset, args.log_wandb)

    # if args.save:
    #     model.save_pretrained(args.save)
