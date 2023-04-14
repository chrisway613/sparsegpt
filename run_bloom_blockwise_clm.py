import os
import gc
import math
import random
import logging
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import datasets
import transformers

from copy import deepcopy
from tqdm.auto import tqdm
from itertools import chain
from datetime import timedelta

from datasets import load_dataset

from torch.utils.data import DataLoader

from accelerate.utils import set_seed
from accelerate.logging import get_logger
from accelerate import Accelerator, InitProcessGroupKwargs

from transformers import default_data_collator
from transformers.models.bloom import BloomConfig, BloomTokenizerFast, BloomForCausalLM

from sparsegpt import SparseGPT
from modelutils import find_layers


logger = get_logger(__name__)


class StateStdout:
    """ A context manager for print out some messages of current state."""

    def __init__(self, logger=None, begin="Start", end="Done"):
        self.logger = logger or get_logger("stdout")

        self.begin = begin
        self.end = end

    def __enter__(self):
        self.logger.info(f"{self.begin}")
        return self

    def __exit__(self, *exc):
        self.logger.info(f"{self.end}\n")


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


def disable_torch_init():
    """ Disable initialization of Pytorch. """

    def skip(*args, **kwargs):
        pass

    torch.nn.init.normal_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.kaiming_uniform_ = skip


@torch.no_grad()
def sequential_eval(
    model: nn.Module, dataset, dataloader, accelerator: Accelerator,
    hard_sparse_weight: bool = False, sparse_ratio: float = 0., verbose: bool = True
):
    """ Evaluation in sequential forwarding way. """

    mode = model.training
    use_cache = model.config.use_cache
    model.config.use_cache = False

    model.eval()

    # Embedding layer
    word_embed = model.transformer.word_embeddings
    word_embed_ln = model.transformer.word_embeddings_layernorm
    # All BloomBlocks
    layers = model.transformer.h
    # LayerNorm
    ln_f = model.transformer.ln_f
    # Language head
    lm_head = model.lm_head

    dev = accelerator.device
    dtype = next(iter(model.parameters())).dtype
    hidden_size = model.config.hidden_size

    losses = []
    loss_fct = nn.CrossEntropyLoss()

    for batch in tqdm(dataloader, disable=not accelerator.is_local_main_process):
        ''' CPU -> GPU '''

        # Embedding layer & the 1st Layer Norm
        model.transformer.word_embeddings = word_embed = accelerator.prepare(word_embed)
        model.transformer.word_embeddings_layernorm = word_embed_ln = accelerator.prepare(word_embed_ln)
        # The 1st BloomBlock
        layers[0] = accelerator.prepare(layers[0])

        ''' Collect output hidden states from embedding layer '''

        batch_size, seq_length = batch['input_ids'].shape[:2]
        # 记录每个样本在输入 BloomBlock 前的 hidden state
        hs = torch.zeros(
            (batch_size, seq_length, hidden_size),
            dtype=dtype, device=dev
        )
        # 记录当前 forward 到第几个 BloomBlock 以及对应的一些状态
        cache = {'attention_mask': None, 'alibi': None}
        # 以下会仅经过了：Embedding 层 -> LayerNorm -> the 1st BloomBlock
        # 从而将每个样本在输入第一个 BloomBlock 前的 hidden state 记录下来
        layers[0] = BatchCatcher(layers[0], cache, hs)
        try:
            model(**batch)
        except ValueError:
            pass

        # GPU -> CPU
        layers[0] = accelerator.unwrap_model(layers[0].module).cpu()
        model.transformer.word_embeddings = word_embed = accelerator.unwrap_model(word_embed).cpu()
        model.transformer.word_embeddings_layernorm = word_embed_ln = accelerator.unwrap_model(word_embed_ln).cpu()

        # Used for all layers
        alibi = cache.pop('alibi')
        attention_mask = cache.pop('attention_mask')

        del cache
        accelerator.clear()

        # Sequential forwarding
        for i in range(len(layers)):
            # CPU -> GPU
            layer = accelerator.prepare(layers[i])
            # 将稀疏的部分置 0
            if hard_sparse_weight:
                subset = find_layers(layer)
                for name in subset:
                    W = subset[name].weight.data
                    # 例如稀疏率是75%，那么先有小到大排序，然后将前 75% 的参数值置0
                    thresh = torch.sort(torch.abs(W.flatten()))[0][int(W.numel() * sparse_ratio)]
                    W.data[torch.abs(W.data) <= thresh] = 0

            # Regard output hidden states as the input hidden states of next layer
            hs = layer(hs, attention_mask=attention_mask, alibi=alibi)[0]

            # GPU -> CPU
            layers[i] = accelerator.unwrap_model(layer).cpu()
            del layer
            accelerator.clear()
        
        # CPU -> GPU
        model.transformer.ln_f = ln_f = accelerator.prepare(ln_f)
        model.lm_head = lm_head = accelerator.prepare(lm_head)

        # Force to fp32 for stability
        logits = lm_head(ln_f(hs)).to(torch.float32)
        # Current token predict the next one
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch['labels'][..., 1:].contiguous()

        # Loss
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        del logits, shift_logits, shift_labels

        # GPU -> CPU
        model.transformer.ln_f = ln_f = accelerator.unwrap_model(ln_f).cpu()
        model.lm_head = lm_head = accelerator.unwrap_model(lm_head).cpu()
        # (num_devices*batch_size,)
        losses.append(accelerator.gather(loss.repeat(batch_size)).cpu())

        del loss
        accelerator.clear()

    losses = torch.cat(losses)
    losses = losses[:len(dataset)]
    try:
        perplexity = math.exp(torch.mean(losses))
    except OverflowError:
        perplexity = float("inf")
    
    del losses
    if verbose:
        logger.info(f"Perplexity: {perplexity}\n")

    model.config.use_cache = use_cache
    model.train(mode=mode)

    return perplexity


# Sparse mask
MASK = {}


@torch.no_grad()
def layer_prune(
    layer: nn.Module, sparsity,
    hidden_states, alibis, attention_masks,
    accelerator: Accelerator, **kwargs
):
    """ Layer-wise pruning by per-sample. """

    mode = layer.training
    layer.eval()

    dev = accelerator.device
    dtype = next(iter(layer.parameters())).dtype
    num_samples = len(hidden_states)

    gpts = {}
    prune_target = kwargs.pop('prune_target', '')
    # 返回一个字典，找出当前 BloomBlock 下的所有 Linear 层
    subset = find_layers(unwrapped_layer)
    for name in subset:
        if prune_target in name:
            gpts[name] = SparseGPT(subset[name])

    def add_batch(name):
        def tmp(_, inp, out):
            # inp 是 tuple，取第一个代表 input_ids
            gpts[name].add_batch(inp[0].data, out.data)
        return tmp

    handles = [subset[name].register_forward_hook(add_batch(name)) for name in gpts]
    for sample_i in tqdm(range(num_samples), desc="Running hooks of pruning", disable=not accelerator.is_local_main_process):
        # 其实这里的目的并非记录每个样本经过当前 BloomBlock 输出后的 hidden state
        # (真正的记录过程在后面)
        # 而是为了 SparseGPT() 做 add_batch()，让前面的注册的 hook 发挥作用
        unwrapped_layer(
            hidden_states[sample_i].unsqueeze(0).to(dev),
            attention_mask=attention_masks[sample_i].unsqueeze(0).to(dev),
            alibi=alibis[sample_i].unsqueeze(0).to(dev)
        )
    for h in handles:
        h.remove()

    sparse_thresh = kwargs.pop('sparse_thresh', 1e-10)
    # 对当前 BloomBlock 中的每个 Linear 层做 pruning
    for name in gpts:
        logger.info(f"Module: {name}")
        gpts[name].fasterprune(sparsity, **kwargs)
        # Binary sparse mask (0|1)
        MASK[name] = (subset[name].weight.abs() > sparse_thresh).to(dtype=dtype, device='cpu')
    del gpts, subset

    layer.train(mode=mode)


def parse_args():
    parser = argparse.ArgumentParser(description="Running BLOOM model on Causal Language Modeling task.")

    # Model
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models."
    )
    parser.add_argument(
        "--model_cache_dir",
        type=str,
        default=None,
        help="Directory of model weights to be cached in."
    )

    # Data
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="The name of the dataset to use (via the datasets library)."
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library)."
    )
    parser.add_argument(
        "--data_cache_dir",
        type=str,
        default=None,
        help="Directory of the dataset to be cached.",
    )
    
    # Train
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        required=True,
        help="Number of training epochs."
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Max sequence length of a data sample."
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup training."
    )

    # Prune
    parser.add_argument(
        "--sparse",
        action="store_true",
        help="Do pruning."
    )
    parser.add_argument(
        '--num_prune_samples', type=int, default=128,
        help='Number of calibration data samples for pruning.'
    )
    parser.add_argument(
        "--per_device_prune_batch_size",
        type=int,
        default=128,
        help="Batch size (per device) for the pruning dataloader.",
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--sparsities',
        nargs="*",
        type=float,
        help='Target sparsities.'
    )
    parser.add_argument(
        '--sparse_steps',
        nargs="*",
        type=int,
        help='Step that execute pruning.'
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
        '--min_layer', type=int, default=-1,
        help='Prune all layers with id >= this.'
    )
    parser.add_argument(
        '--max_layer', type=int, default=1000,
        help='Prune all layers with id < this.'
    )
    parser.add_argument(
        '--prune_only', type=str, default='',
        help='Prune only layers that contain this text.'
    )
    parser.add_argument(
        "--path_to_dense",
        type=str,
        default="bloom-dense",
        help="Path to dense weigths."
    )
    parser.add_argument(
        "--dense_metric",
        type=float,
        default=0.,
        help="Metric of dense, used for evaluating sparse model."
    )

    # Global
    parser.add_argument(
        '--eval_dense',
        action='store_true',
        help="Do evaluation for dense model."
    )
    parser.add_argument(
        "--dump_dense",
        action="store_true",
        help="Record dense weights to a specified path."
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Only do evaluation."
    )
    parser.add_argument(
        "--eval_full_data",
        action="store_true",
        help="If not true, only eval on 1000 samples of validation data."
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Whether to turn on debug mode. If true, only 128 samples of training data will be selected.'
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible training."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="bloom-outputs",
        help="Where to store the results."
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Parse arguments
    args = parse_args()
    logger.info(f"Target sparsities: {args.sparsities}\n")

    # Accelerator(for distributed training)
    kwargs_handlers = [InitProcessGroupKwargs(timeout=timedelta(seconds=180000))]
    accelerator = Accelerator(kwargs_handlers=kwargs_handlers, gradient_accumulation_steps=args.gradient_accumulation_steps)
    
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Random seed
    set_seed(args.seed)

    # Debug mode
    if args.debug:
        # Only 16 samples will be used to debug
        args.num_prune_samples = 16
        args.per_device_prune_batch_size = 16
    # Normal train/eval
    else:
        # Make output directory if not existed
        os.makedirs(args.output_dir, exist_ok=True)

    accelerator.wait_for_everyone()

    # Load dataset
    with StateStdout(logger=logger, begin="Loading datset", end="Done"):
        data = load_dataset(
            args.dataset_name, args.dataset_config_name,
            cache_dir=args.data_cache_dir, use_auth_token=True
        )
        if 'validation' not in data:
            # val : train = 5% : 95%
            num_val_samples = int(0.05 * len(data['train']))
            val_indices = random.sample(range(len(data['train'])), num_val_samples)
            train_indices = [i for i in range(len(data['train'])) if i not in val_indices]
            assert len(train_indices) + len(val_indices) == len(data['train'])

            data['validation'] = data['train'].select(val_indices)
            data['train'] = data['train'].select(train_indices)

    # Tokenize data texts
    tokenizer = BloomTokenizerFast.from_pretrained(args.model_name_or_path, cache_dir=args.model_cache_dir)

    column_names = data['train'].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    with accelerator.main_process_first():
        data = data.map(
            lambda samples: tokenizer(samples[text_column_name]),
            batched=True,
            num_proc=8,
            remove_columns=column_names,
            load_from_cache_file=True,
            desc="Running tokenizer on dataset"
        )
    
    if args.max_seq_length > tokenizer.model_max_length:
        args.max_seq_length = tokenizer.model_max_length
        logger.warning(
            f"The max_seq_length passed ({args.max_seq_length}) is larger than the maximum length for the model"
            f"({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}.\n"
        )
    
    # Processing data
    def group_texts(examples):
        """ Concatenate all texts from dataset and generate chunks of max_sequence_length. """

        # Concatenate all texts
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop
        if total_length >= args.max_seq_length:
            total_length = (total_length // args.max_seq_length) * args.max_seq_length

        # Split by chunks of max_seq_length
        result = {
            k: [t[i: i + args.max_seq_length]
                for i in range(0, total_length, args.max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()

        return result

    with accelerator.main_process_first():
        data = data.map(
            group_texts,
            batched=True,
            num_proc=8,
            load_from_cache_file=True,
            desc=f"Grouping texts in chunks of {args.max_seq_length}",
        )
    
    # Data split
    val_data = data['validation']
    if not args.eval_full_data and len(val_data) > 1280:
        data['validation'] = data['validation'].select(range(1280))
        logger.info("NOTE: only 1280 samples of validation data selected.")
    if not args.eval_only:
        train_data = data['train']
    if args.debug:
        train_data = data['train'].select(range(96))
        # NOTE: debug=overfit
        val_data = train_data
        logger.info(f"NOTE: debug mode on! only 96 train(eval) data samples selected.\n")
    
    # Let u know the number of samples, no need to thx to me ~
    logger.info(f"\tNum validation data = {len(val_data)}")
    if not args.eval_only:
        logger.info(f"\tNum training data = {len(train_data)}")
    logger.info("\n")

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_data)), 3):
        logger.info(f"Sample {index} of the training data: {train_data[index]}.")
    
    # Dataloader
    eval_dataloader = DataLoader(
        val_data, batch_size=args.per_device_eval_batch_size,
        collate_fn=default_data_collator, pin_memory=True, num_workers=8
    )
    if not args.eval_only:
        # NOTE: we should not shuffle here.
        # Cuz the target of model is aligning with teacher, not the data label
        train_dataloader = DataLoader(
            train_data, batch_size=args.per_device_train_batch_size,
            collate_fn=default_data_collator, pin_memory=True, num_workers=8
        )
    
    # Model
    # NOTE: this is inherit from sparsegpt, but I don't know the reason
    disable_torch_init()

    model_config = BloomConfig.from_pretrained(args.model_name_or_path, cache_dir=args.model_cache_dir)
    model = BloomForCausalLM.from_pretrained(
        args.model_name_or_path,
        config=model_config,
        # torch_dtype='auto',  # If set, and in fp16 training, the model params dtype will be fp16.
        cache_dir=args.model_cache_dir
    )
    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    logger.info(f"\nModel structure:\n{model}\n")

    # Data parallel
    eval_dataloader = accelerator.prepare(eval_dataloader)
    if not args.eval_only:
        train_dataloader = accelerator.prepare(train_dataloader)

    # Dense model evaluation
    if args.eval_dense:
        with StateStdout(logger=logger, begin="Dense model eval .."):
            args.dense_metric = sequential_eval(model, val_data, eval_dataloader, accelerator)

    # Log global messages
    per_device_batch_size = args.per_device_train_batch_size \
        if not args.eval_only else args.per_device_eval_batch_size
    total_batch_size = per_device_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_examples = len(train_data) if not args.eval_only else len(val_data)
    if not args.eval_only:
        # num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        # total_steps = num_update_steps_per_epoch * args.num_train_epochs
        total_steps = args.num_train_epochs

    logger.info("\n***** BEGIN *****")
    logger.info(f"  Num examples = {num_examples}")
    logger.info(f"  Instantaneous batch size per device = {per_device_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    if not args.eval_only:
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Total optimization steps = {total_steps}")
        # logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info("\n")

    # Eval only
    if args.eval_only:
        with StateStdout(logger=logger, begin="NOTE: Program for Eval only"):
            sequential_eval(model, val_data, eval_dataloader, accelerator)
    # Train(& Prune)
    else:
        # Customize learning rate scheduler
        def lr_schedule(step):
            if step < args.num_warmup_steps:
                return float(step) / float(max(1, args.num_warmup_steps))
            return max(0.0, float(total_steps - step)) / float(max(1, total_steps - args.num_warmup_steps))

        # NOTE: We only train BloomBlocks
        model.transformer.word_embeddings.requires_grad_(False)
        model.transformer.word_embeddings_layernorm.requires_grad_(False)
        model.transformer.ln_f.requires_grad_(False)
        model.lm_head.requires_grad_(False)

        hidden_size = model.config.hidden_size
        dtype = next(iter(model.parameters())).dtype

        use_cache = model.config.use_cache
        model.config.use_cache = False

        model.train()

        # Word embedding layer
        word_embed = model.transformer.word_embeddings
        word_embed_ln = model.transformer.word_embeddings_layernorm
        # BloomBlocks
        layers = model.transformer.h

        # Output hidden states from previous dense layer
        hs_batches = []
        # Output hidden states from teacher
        out_batches = []
        alibi_batches = []
        attention_mask_batches = []

        # Output hidden states from previous dense layer for pruning
        prune_hidden_states = []
        prune_alibis = []
        prune_attention_masks = []

        # Current cuda device
        dev = accelerator.device

        # Block-wise forwarding, i.e. 
        # It is turn to the next block only after the previous block done with all batches
        for layer_i, layer in enumerate(layers):
            # CPU -> GPU
            layer = layers[layer_i] = accelerator.prepare(layer)
            unwrapped_layer = accelerator.unwrap_model(layer)

            msg = "Collecting output hidden states from teacher .."
            with StateStdout(logger=logger, begin=msg):
                layer.eval()

                with torch.no_grad():
                    out_batches.clear()
                    for batch_i, batch in tqdm(
                        enumerate(train_dataloader), desc="Collect Info",
                        disable=not accelerator.is_local_main_process
                    ):
                        # The 1st layer need to collect outputs from embedding layer
                        if layer_i == 0:
                            ''' CPU -> GPU '''

                            model.transformer.word_embeddings = word_embed = accelerator.prepare(word_embed)
                            model.transformer.word_embeddings_layernorm = word_embed_ln = accelerator.prepare(word_embed_ln)

                            # Outputs from embedding layer
                            hs = torch.zeros(
                                (per_device_batch_size, args.max_seq_length, hidden_size),
                                dtype=dtype, device=dev
                            )
                            # 记录当前 forward 到第几个 BloomBlock 以及对应的一些状态
                            cache = {'attention_mask': None, 'alibi': None}
                            # 以下会仅经过了：Embedding 层 -> LayerNorm -> the 1st BloomBlock
                            # 从而将每个样本在输入第一个 BloomBlock 前的 hidden state 记录下来
                            layers[layer_i] = layer = BatchCatcher(layer, cache, hs)
                            # BloomBlock 前的部分(Embedding 层)不需要梯度
                            with torch.no_grad():
                                try:
                                    model(**batch)
                                except ValueError:
                                    pass
                            
                            ''' GPU -> CPU '''

                            # Collect outputs from embedding layer
                            hs_batches.append(hs.cpu())
                            alibi_batches.append(cache.pop('alibi').cpu())
                            attention_mask_batches.append(cache.pop('attention_mask').cpu())

                            # Extract layer from BatchCacher
                            layers[layer_i] = layer = layer.module
                            
                            ''' GPU -> CPU '''

                            model.transformer.word_embeddings = word_embed = accelerator.unwrap_model(word_embed).cpu()
                            model.transformer.word_embeddings_layernorm = word_embed_ln = accelerator.unwrap_model(word_embed_ln).cpu()
                        
                            # Pop out embedding layer
                            accelerator._models.pop()
                            accelerator._models.pop()

                        ''' CPU -> GPU '''

                        # Output hidden states from previous dense layer
                        hs = hs_batches[batch_i].to(dev)
                        alibi = alibi_batches[batch_i].to(dev)
                        attention_mask = attention_mask_batches[batch_i].to(dev)

                        # [BE CAREFUL] detach and thus cut out the relationship between layers
                        out = layer(hs, attention_mask=attention_mask, alibi=alibi)[0].detach().cpu()
                        out_batches.append(out)

                        ''' CPU -> GPU '''

                        hs_batches[batch_i] = hs.cpu()
                        alibi_batches[batch_i] = alibi.cpu()
                        attention_mask_batches[batch_i] = attention_mask.cpu()

                        del out, alibi, attention_mask

                        ''' Release memories '''

                        del hs
                        gc.collect()
                        torch.cuda.empty_cache()
            
                layer.train()

            # NOTE: here, step means number of epochs
            completed_step = 0
            completed_prune_times = 0

            total_train_steps = args.num_train_epochs * len(train_dataloader)
            # Only show the progress bar once on each machine.
            progress_bar = tqdm(range(total_train_steps), desc=f"Train layer {layer_i + 1}", disable=not accelerator.is_local_main_process)

            # Init lr
            lr = args.learning_rate

            # Optimizer
            optimizer = accelerator.prepare(torch.optim.AdamW(unwrapped_layer.parameters(), lr=lr))
            # NOTE: Set initial gradients to None can save some memories
            optimizer.zero_grad(set_to_none=True)

            for epoch in range(args.num_train_epochs):
                # Pruning
                if args.sparse and completed_prune_times < len(args.sparsities) \
                    and (args.min_layer <= layer_i < args.max_layer) and \
                        completed_step == args.sparse_steps[completed_prune_times]:
                    sparsity = args.sparsities[completed_prune_times]
                    
                    ''' Random select pruning samples(hidden states) from training samples(hidden states) '''

                    num_prune_batches = min(args.num_prune_samples // hs_batches[0].size(0), len(hs_batches))
                    prune_batch_indices = random.sample(range(len(hs_batches)), num_prune_batches)
                    
                    # (num_prune_batches * train_batch_size, seq_length, hidden_size)
                    prune_hidden_states = torch.cat([hs_batches[batch_i] for batch_i in prune_batch_indices])
                    prune_alibis = torch.cat([alibi_batches[batch_i] for batch_i in prune_batch_indices])
                    prune_attention_masks = torch.cat([attention_mask_batches[batch_i] for batch_i in prune_batch_indices])
                    logger.info(f"{prune_hidden_states.size(0)} pruning samples selected.")

                    with StateStdout(logger=logger, begin=f"Pruning layer {layer_i + 1} to sparsity {sparsity} .."):
                        layer_prune(
                            unwrapped_layer, sparsity,
                            prune_hidden_states, prune_alibis, prune_attention_masks,
                            accelerator, prune_target = args.prune_only
                        )

                    completed_prune_times += 1
                    del prune_hidden_states, prune_alibis, prune_attention_masks
                    gc.collect()
                    
                    # Eval after pruning
                    with StateStdout(logger=logger, begin=f"Eval after layer {layer_i + 1} pruned .."):
                        ppl = sequential_eval(
                            model,
                            val_data, eval_dataloader, accelerator,
                            hard_sparse_weight=args.gmp, sparse_ratio=sparsity
                        )

                        # NOTE: 'sequential_eval()' will Release all memories,
                        # so we should put layer back to proper device
                        layer = layers[layer_i] = accelerator.prepare(layers[layer_i])
                        unwrapped_layer = accelerator.unwrap_model(layer)

                        if ppl <= args.dense_metric:
                            logger.info(f"NOTE: The perplexity({ppl}) is better than dense's({args.dense_metric}), skip training for this layer.\n")
                            break

                # Loss among all data
                layer_loss = 0.
                # Cosine similarity(vs teacher) among all data
                layer_similarity = 0.

                # Train loop
                for batch_i in range(len(train_dataloader)):
                    ''' CPU -> GPU '''

                    # Output hidden states from previous dense layer
                    hs = hs_batches[batch_i].to(dev)
                    alibi = alibi_batches[batch_i].to(dev)
                    attention_mask = attention_mask_batches[batch_i].to(dev)

                    ''' Align with teacher '''

                    # Forward
                    out = layer(hs, attention_mask=attention_mask, alibi=alibi)[0]

                    ''' CPU -> GPU '''

                    hs_batches[batch_i] = hs.cpu()
                    alibi_batches[batch_i] = alibi.cpu()
                    attention_mask_batches[batch_i] = attention_mask.cpu()
                    del hs, alibi, attention_mask

                    # Norm before computing loss, this is for stability
                    var = out.pow(2).mean(dim=-1, keepdim=True)
                    teacher_out = out_batches[batch_i].to(dev)
                    teacher_var = teacher_out.pow(2).mean(dim=-1, keepdim=True)

                    # Loss
                    loss = F.mse_loss(
                        out / (var + 1e-6),
                        teacher_out / (teacher_var + 1e-6)
                    )
                    raw_loss = loss.item()
                    layer_loss += raw_loss / len(train_dataloader)

                    # Gradient accumulation
                    loss /= len(train_dataloader)
                    # If mixed precision training, accelerator will scale loss here
                    accelerator.backward(loss)
                    del loss, var, teacher_var

                    # Cosine similarity between student & teacher
                    # No need for gradients
                    with torch.no_grad():
                        sim = F.cosine_similarity(
                            out.reshape(-1, hidden_size),
                            teacher_out.reshape(-1, hidden_size)
                        ).mean()
                    # Reduce across all process
                    sim = (accelerator.reduce(sim) / accelerator.num_processes).item()
                    layer_similarity += sim / len(train_dataloader)
                    del out

                    gc.collect()
                    torch.cuda.empty_cache()

                    # Log training info
                    if batch_i % 10 == 0:
                        logger.info(f"epoch {epoch + 1}\t batch {batch_i + 1}\t loss {raw_loss}\t similarity {sim}\n")

                    progress_bar.update()

                # Update parameters
                optimizer.step()
                # [BE CAREFUL] Clear accumulated gradients if parameters updated
                optimizer.zero_grad(set_to_none=True)
                lr = optimizer.param_groups[0]['lr']

                # Epoch counted
                completed_step += 1
                # Update lr
                next_lr = args.learning_rate * lr_schedule(completed_step)
                for group in optimizer.param_groups:
                    group['lr'] = next_lr
                
                # Apply sparse mask
                if MASK:
                    subset = find_layers(unwrapped_layer)
                    for name, module in subset.items():
                        mask = MASK[name].to(device=module.weight.device, dtype=module.weight.dtype)
                        module.weight.data = module.weight.data * mask
                        MASK[name] = mask.cpu()

                # Eval when a epoch done
                with StateStdout(logger=logger, begin=f"Eval in epoch {epoch + 1} .."):
                    ppl = sequential_eval(
                        model,
                        val_data, eval_dataloader, accelerator, verbose=False,
                        hard_sparse_weight=args.gmp, sparse_ratio=(sparsity if completed_prune_times else 0.)
                    )

                    # NOTE: 'sequential_eval()' will Release all memories,
                    # so we should put layer back to proper device
                    layer = layers[layer_i] = accelerator.prepare(layers[layer_i])
                    unwrapped_layer = accelerator.unwrap_model(layer)
                
                # Log training info when the epoch done
                logger.info(
                    f"Layer {layer_i + 1}\t Epoch {epoch + 1}\t lr {lr}\t next lr {next_lr}\n"
                    f"Loss {layer_loss}\t Similarity {layer_similarity}\t Perplexity {ppl}\n"
                )
                
                ''' Release memories '''

                gc.collect()
                torch.cuda.empty_cache()

                # Done if it is close enough to teacher
                if layer_similarity > 0.998:
                    logger.info(f"NOTE: Similarity > 0.998, layer {layer_i + 1} is DONE!\n")
                    break
                if ppl <= args.dense_metric:
                    logger.info(f"NOTE: The perplexity({ppl}) is better than dense's({args.dense_metric}), layer {layer_i + 1} is DONE!\n")
                    break
            
            # Clear sparse mask, cuz it is no use for next layer
            MASK.clear()

            # Set input hidden states for next layer
            # Clear output hidden states from teacher(and will be set later, don't worry)
            hs_batches, out_batches = out_batches, []

            # GPU -> CPU
            layers[layer_i] = unwrapped_layer.to('cpu')
            del unwrapped_layer, layer, optimizer
            # Releases all references to the internal objects stored and call the garbage collector
            accelerator.clear()
            
        model.config.use_cache = use_cache

        del hs_batches, out_batches, alibi_batches, attention_mask_batches
        del prune_hidden_states, prune_alibis, prune_attention_masks
        gc.collect()

        # Save results if not in debug mode.
        if not args.debug:
            accelerator.wait_for_everyone()
            model.save_pretrained(
                args.output_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)

    # Releases all references to the internal objects stored and call the garbage collector.
    accelerator.clear()
