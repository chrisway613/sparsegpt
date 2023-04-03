import time
import argparse

import torch
import torch.nn as nn

from tqdm import tqdm

from sparsegpt import SparseGPT
from datautils import get_loaders
from modelutils import find_layers


try:
    import wandb
    has_wandb = True
except:
    has_wandb = False    


def get_bloom(model):
    import torch
    
    def skip(*args, **kwargs):
        pass

    # TODO: what do these attempt to do?
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    from transformers import BloomForCausalLM
    
    model = BloomForCausalLM.from_pretrained(model, cache_dir='/ssd1/models/bloom', torch_dtype='auto')
    model.seqlen = 2048
    
    return model


@torch.no_grad()
def bloom_sequential(model, dataloader, dev, means=None, stds=None):
    print('Starting..\n')

    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.transformer.h

    model.transformer.word_embeddings = model.transformer.word_embeddings.to(dev)
    model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.to(dev)
    # The 1st BloomBlock
    layers[0] = layers[0].to(dev)

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
            model(batch[0].to(dev))
        except ValueError:
            pass

    # layers[0] = layers[0].module
    # layers[0] = layers[0].cpu()
    layers[0] = layers[0].module.cpu()
    model.transformer.word_embeddings = model.transformer.word_embeddings.cpu()
    model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.cpu()
    
    torch.cuda.empty_cache()

    # 记录每个样本经过 BloomBlock 输出后的 hidden states
    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    alibi = cache['alibi']

    print('Ready!\n')

    for i in range(len(layers)):
        layer = layers[i].to(dev)
        # 返回一个字典，找出当前 BloomBlock 下的所有 Linear 层
        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            if (not (args.minlayer <= i < args.maxlayer and args.prune_only in name)) == (not args.invert):
                continue

            gpts[name] = SparseGPT(subset[name])

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
            # outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, alibi=alibi)[0]
            layer(inps[j].unsqueeze(0), attention_mask=attention_mask, alibi=alibi)

        for h in handles:
            h.remove()

        # 对当前 BloomBlock 中的每个 Linear 层做 pruning
        print('Pruning..')
        for name in gpts:
            print(f"layer: {i}\tname: {name}")

            gpts[name].fasterprune(
                args.sparsity,
                prunen=args.prunen,
                prunem=args.prunem,
                percdamp=args.percdamp
            )
        print("Done!\n")

        # Pruning 后，记录每个样本经过当前 BloomBlock 输出后的 hidden state
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, alibi=alibi)[0]

        layers[i] = layer.cpu()

        del gpts
        # TODO: verify this
        del layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache


@torch.no_grad()
def bloom_eval(model, testenc, dev, dataset: str, log_wandb: bool = False):
    print('Evaluation..\n')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.transformer.h

    model.transformer.word_embeddings = model.transformer.word_embeddings.to(dev)
    model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.to(dev)
    layers[0] = layers[0].to(dev)

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
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass

    # layers[0] = layers[0].module
    # layers[0] = layers[0].cpu()
    layers[0] = layers[0].module.cpu()
    model.transformer.word_embeddings = model.transformer.word_embeddings.cpu()
    model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.cpu()

    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    alibi = cache['alibi']

    for i in range(len(layers)):
        layer = layers[i].to(dev)

        # 将稀疏的部分置0
        if args.gmp:
            subset = find_layers(layer)
            for name in subset:
                W = subset[name].weight.data
                # 例如稀疏率是75%，那么先有小到大排序，然后将前 75% 的参数值置0
                thresh = torch.sort(torch.abs(W.flatten()))[0][int(W.numel() * args.sparsity)]
                W.data[torch.abs(W.data) <= thresh] = 0

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, alibi=alibi)[0]

        layers[i] = layer.cpu()

        del layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.transformer.ln_f = model.transformer.ln_f.to(dev)
    model.lm_head = model.lm_head.to(dev)

    # TODO: pls see line 260
    # testenc = testenc.to(dev)
    loss_fct = nn.CrossEntropyLoss()

    nlls = []
    for i in tqdm(range(nsamples)):
        hidden_states = inps[i].unsqueeze(0)
        # hidden_states = model.transformer.ln_f(hidden_states)
        # lm_logits = model.lm_head(hidden_states)
        lm_logits = model.lm_head(model.transformer.ln_f(hidden_states))

        shift_logits = lm_logits[:, :-1, :].contiguous()
        # shift_labels = testenc[
        #     :, (i * model.seqlen):((i + 1) * model.seqlen)
        # ][:, 1:]
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:].to(dev)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
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
        '--seed',
        type=int, default=42, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
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
        '--debug',
        action='store_true',
        help='Whether to turn on debug mode. If true, only 1000 samples will be selected for training data.'
    )

    args = parser.parse_args()
    # init W&B logging
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

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples,
        seed=args.seed, model=args.model,
        seqlen=model.seqlen, debug=args.debug
    )

    if args.eval_dense:
        print("[Eval for dense]")
        bloom_eval(model, testloader, DEV, args.dataset, args.log_wandb)

    if (args.sparsity or args.prunen) and not args.gmp:
        bloom_sequential(model, dataloader, DEV)
        # for n, p in model.named_parameters():
        #     print(f"name: {n}\tsparsity: {torch.mean((p == 0).float())}")
        #     # 仅看第一个 BloomBlock，因为后面都是同样的情况
        #     if 'dense_4h_to_h' in n:
        #         break
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and 'head' not in name:
                print(f"name: {name}\tsparsity: {torch.mean((module.weight == 0).float())}")

    # for dataset in ['wikitext2', 'ptb', 'c4']:
        # _, testloader = get_loaders(
        #     dataset, seed=args.seed,
        #     model=args.model, seqlen=model.seqlen
        # )
        # print(f"Dataset: {dataset}")
        # bloom_eval(model, testloader, DEV, dataset, args.log_wandb)

    bloom_eval(model, testloader, DEV, args.dataset, args.log_wandb)
    if args.save:
        model.save_pretrained(args.save)
