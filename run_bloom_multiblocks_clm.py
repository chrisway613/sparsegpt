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

from tqdm.auto import tqdm
# from itertools import chain
from datetime import timedelta

from datasets import load_dataset

from torch.utils.data import DataLoader, Subset

from accelerate.utils import set_seed
from accelerate.logging import get_logger
from accelerate import Accelerator, InitProcessGroupKwargs

from transformers import default_data_collator, SchedulerType, get_scheduler
from transformers.models.bloom import BloomConfig, BloomTokenizerFast, BloomForCausalLM

from transformers.models.bloom.modeling_bloom import BloomBlock

# from peft import LoraConfig, TaskType, get_peft_model

from modelutils import find_layers
from sparsegpt import SparseGPT, ABCSolver
from chatllm_data_utils import create_prompt_dataset


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


class MultiBloomBlocks(nn.Sequential):    
    def forward(self, hidden_states: torch.Tensor, alibi: torch.Tensor, attention_mask: torch.Tensor):
        for module in self:
            # NOTE: The output from BloomBlock is tuple, so we use subscript [0] for extracting hidden states
            hidden_states = module(hidden_states, alibi, attention_mask)[0]
        # Return tuple for consistent with the output form of a single block
        return (hidden_states,)


def disable_torch_init():
    """ Disable initialization of Pytorch. """

    def skip(*args, **kwargs):
        pass

    torch.nn.init.normal_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.kaiming_uniform_ = skip


@torch.no_grad()
def sequential_eval(
    model: nn.Module, num_layers_aggregated, dataset, dataloader, accelerator: Accelerator,
    hard_sparse_weight: bool = False, sparse_ratio: float = 0., verbose: bool = True
):
    """ Evaluation in sequential forwarding way. """

    mode = model.training
    use_cache = model.config.use_cache
    model.config.use_cache = False
    model.eval()

    # All BloomBlocks
    layers = model.transformer.h

    dev = accelerator.device
    dtype = next(iter(model.parameters())).dtype
    hidden_size = model.config.hidden_size

    losses = []
    loss_fct = nn.CrossEntropyLoss()

    for batch in tqdm(dataloader, disable=not accelerator.is_local_main_process):
        ''' CPU -> GPU '''

        model.transformer.word_embeddings.to(dev)
        model.transformer.word_embeddings_layernorm.to(dev)
        layers[0].to(dev)

        ''' Collect output hidden states from embedding layer '''

        batch_size, seq_length = batch['input_ids'].shape[:2]
        # 记录每个样本在输入 BloomBlock 前的 hidden state
        hs = torch.zeros(
            (batch_size, seq_length, hidden_size),
            dtype=dtype, device='cpu'
        )
        
        cache = {'layer_i': 0, 'attention_mask': None, 'alibi': None}
        layers[0] = BatchCatcher(layers[0], cache, hs)
        
        try:
            model(**batch)
        except ValueError:
            pass

        ''' GPU -> CPU '''

        layers[0] = layers[0].module.cpu()
        model.transformer.word_embeddings.cpu()
        model.transformer.word_embeddings_layernorm.cpu()

        # Used for all layers
        alibi = cache.pop('alibi')
        attention_mask = cache.pop('attention_mask')
        del cache

        for layer_start in tqdm(range(0, len(layers), num_layers_aggregated), desc="multi-layers forwarding", disable=not accelerator.is_local_main_process):
            layer_end = min(layer_start + num_layers_aggregated, len(layers))
            
            sub_module = MultiBloomBlocks(*layers[layer_start:layer_end])
            sub_module.to(dev)

            hs = sub_module(hs.to(dev), attention_mask=attention_mask, alibi=alibi)[0]
            
            sub_module.cpu()
            del sub_module
        
        ''' CPU -> GPU '''

        model.transformer.ln_f.to(dev)
        model.lm_head.to(dev)

        batch_loss = []
        for hs_sample, label in tqdm(zip(hs, batch['labels']), desc="per-sample counting loss", disable=not accelerator.is_local_main_process):
            logits = model.lm_head(model.transformer.ln_f(hs_sample.unsqueeze(0).to(dev)))
            # Current token predict the next one
            logits = logits[:, :-1, :].contiguous()
            shift_labels = label.unsqueeze(0)[..., 1:].contiguous()
            
            # Loss
            loss = loss_fct(logits.view(-1, logits.size(-1)), shift_labels.view(-1))
            batch_loss.append(loss)
            del logits, shift_labels, loss

        ''' GPU -> CPU '''

        model.lm_head.cpu()
        model.transformer.ln_f.cpu()

        # (num_devices*batch_size,)
        losses.append(accelerator.gather(torch.stack(batch_loss)))
        del batch_loss, hs
        gc.collect()

    losses = torch.cat(losses)
    losses = losses[:len(dataset)]
    try:
        perplexity = math.exp(torch.mean(losses))
    except OverflowError:
        perplexity = float("inf")
    
    if verbose:
        logger.info(f"Perplexity: {perplexity}\n")
    
    del losses
    gc.collect()
    torch.cuda.empty_cache()

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

    pruner = kwargs.pop('pruner', 'sparsegpt')

    dev = accelerator.device
    dtype = next(iter(layer.parameters())).dtype
    num_samples = len(hidden_states)

    mode = layer.training
    layer.eval()

    layer_dev = next(iter(layer.parameters())).device    
    layer.to(dev)

    gpts = {}
    prune_target = kwargs.pop('prune_target', '')
    # 返回一个字典，找出当前 BloomBlock 下的所有 Linear 层
    subset = find_layers(layer, name=kwargs.pop("prefix", ""))
    for name in subset:
        if prune_target in name:
            gpts[name] = SparseGPT(subset[name]) if pruner == 'sparsegpt' else ABCSolver(subset[name])

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
        layer(
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
        if pruner == 'sparsegpt':
            gpts[name].fasterprune(sparsity, **kwargs)
        else:
            gpts[name].prune_structured(sparsity, **kwargs)
        
        # Reduce across all processes.
        subset[name].weight.data = accelerator.reduce(subset[name].weight.data, reduction="mean")
        # Binary sparse mask (0|1)
        MASK[name] = (subset[name].weight.abs() > sparse_thresh).to(dtype=dtype, device='cpu')

    layer.to(layer_dev)
    layer.train(mode=mode)

    del gpts, subset
    gc.collect()
    torch.cuda.empty_cache()


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
    parser.add_argument(
        "--gradient_checkpointing_enable",
        action="store_true",
        help="Activates gradient checkpointing for model."
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
        default=256,
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
        "--max_num_train_samples",
        type=int,
        default=None,
        help="The maximum number of training samples."
    )
    parser.add_argument(
        "--num_train_samples_for_dataloader",
        type=int,
        default=None,
        help="Number of train samples feed to dataloader."
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
    parser.add_argument(
        "--optimizer_type",
        type=str,
        default='adamw',
        help="Type of optimizer."
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts",
                 "polynomial", "constant", "constant_with_warmup"]
    )
    parser.add_argument(
        "--lora",
        action="store_true",
        help="Whether to use Low-Rank decomposition."
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=128,
        help="Lora attention dimension."
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        choices=["no", "fp16", "bf16"],
        default=None,
        help="Whether or not to use mixed precision training (fp16 or bfloat16). Choose from 'no','fp16','bf16'."
    )

    # Prune
    parser.add_argument(
        "--pruner",
        type=str,
        default="sparsegpt",
        help="Pruner type."
    )
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
        "--per_layer_pruning",
        action="store_true",
        help="Whether to prune only one layer(or it can be multiple layers) each time."
    )
    parser.add_argument(
        "--num_layers_aggregated",
        type=int,
        default=1,
        help="Num layers to be aggragated to a module for training each time."
    )
    parser.add_argument(
        "--dense_metric",
        type=float,
        default=None,
        help="Metric of dense, used for evaluating sparse model."
    )
    parser.add_argument(
        "--eval_finetuned_sparse",
        action="store_true",
        help="Eval finetuned sparse model."
    )

    # Global
    parser.add_argument(
        '--eval_dense',
        action='store_true',
        help="Do evaluation for dense model."
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
    accelerator = Accelerator(
        kwargs_handlers=kwargs_handlers,
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    
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

    accelerator.wait_for_everyone()

    # Load dataset
    # with StateStdout(logger=logger, begin="Loading datset", end="Done"):
    #     data = load_dataset(
    #         args.dataset_name, args.dataset_config_name,
    #         cache_dir=args.data_cache_dir, use_auth_token=True
    #     )
    #     if 'validation' not in data:
    #         # val : train = 5% : 95%
    #         num_val_samples = int(0.05 * len(data['train']))
    #         val_indices = random.sample(range(len(data['train'])), num_val_samples)
    #         train_indices = [i for i in range(len(data['train'])) if i not in val_indices]
    #         assert len(train_indices) + len(val_indices) == len(data['train'])

    #         data['validation'] = data['train'].select(val_indices)
    #         data['train'] = data['train'].select(train_indices)

    # Tokenize data texts
    tokenizer = BloomTokenizerFast.from_pretrained(args.model_name_or_path, cache_dir=args.model_cache_dir)

    # column_names = data['train'].column_names
    # text_column_name = "text" if "text" in column_names else column_names[0]

    # with accelerator.main_process_first():
    #     data = data.map(
    #         lambda samples: tokenizer(samples[text_column_name]),
    #         batched=True,
    #         num_proc=8,
    #         remove_columns=column_names,
    #         load_from_cache_file=True,
    #         desc="Running tokenizer on dataset"
    #     )
    
    if args.max_seq_length > tokenizer.model_max_length:
        args.max_seq_length = tokenizer.model_max_length
        logger.warning(
            f"The max_seq_length passed ({args.max_seq_length}) is larger than the maximum length for the model"
            f"({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}.\n"
        )
    
    # Processing data
    # def group_texts(examples):
    #     """ Concatenate all texts from dataset and generate chunks of max_sequence_length. """

    #     # Concatenate all texts
    #     concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    #     total_length = len(concatenated_examples[list(examples.keys())[0]])
    #     # We drop the small remainder, we could add padding if the model supported it instead of this drop
    #     if total_length >= args.max_seq_length:
    #         total_length = (total_length // args.max_seq_length) * args.max_seq_length

    #     # Split by chunks of max_seq_length
    #     result = {
    #         k: [t[i: i + args.max_seq_length]
    #             for i in range(0, total_length, args.max_seq_length)]
    #         for k, t in concatenated_examples.items()
    #     }
    #     result["labels"] = result["input_ids"].copy()

    #     return result

    # with accelerator.main_process_first():
    #     data = data.map(
    #         group_texts,
    #         batched=True,
    #         num_proc=8,
    #         load_from_cache_file=True,
    #         desc=f"Grouping texts in chunks of {args.max_seq_length}",
    #     )
    
    # Data split
    # val_data = data['validation']
    # # TODO: uncomment below
    # # if not args.eval_full_data and len(val_data) > 1280:
    # #     data['validation'] = data['validation'].select(range(1280))
    # #     logger.info("NOTE: only 1280 samples of validation data selected.")
    # # if not args.eval_only:
    # #     train_data = data['train']
    # # TODO: remove future
    # train_data = data['test']
    # val_data = data['test']
    # if args.debug:
    #     train_data = train_data.select(range(96))
    #     # NOTE: debug=overfit
    #     val_data = train_data
    #     logger.info(f"NOTE: debug mode on! only 96 train(eval) data samples selected.\n")
    
    with StateStdout(logger=logger, begin="Loading & processing data.."):
        # Note: The processed data will be saved to a specified directory.
        train_data, val_data = create_prompt_dataset(
            accelerator.local_process_index,
            ["sharegpt:/hdd66/chatllm_datasets/sharegpt-train.json|/hdd66/chatllm_datasets/sharegpt-eval.json"],
            "1,0,0", "./chatllm_data_files", 1, args.seed, tokenizer, args.max_seq_length
        )

        # Select part of training data
        if args.max_num_train_samples:
            if args.max_num_train_samples < len(train_data):
                sample_indices = random.sample(range(len(train_data)), args.max_num_train_samples)
                train_data = Subset(train_data, sample_indices)
                logger.info(f"Note: {args.max_num_train_samples} of training data selected.")
            else:
                logger.warning(
                    f"'max_num_train_samples'({args.max_num_train_samples}) >= "
                    f"{len(train_data)}(which is the total number of training samples), "
                    f"so, use full training dataset instead."
                )

    # Let u know the number of samples, no need to thx to me ~
    logger.info(f"\tNum validation data = {len(val_data)}")
    if not args.eval_only:
        logger.info(f"\tNum training data = {len(train_data)}")

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_data)), 3):
        logger.info(f"Sample {index} of the training data: {train_data[index]}.")
    
    # Dataloader
    eval_dataloader = DataLoader(
        val_data, batch_size=args.per_device_eval_batch_size,
        collate_fn=default_data_collator, pin_memory=True, num_workers=8
    )
    # if not args.eval_only:
    #     # NOTE: we should not shuffle here.
    #     # Cuz the target of model is aligning with teacher, not the data label
    #     train_dataloader = DataLoader(
    #         train_data, batch_size=args.per_device_train_batch_size,
    #         collate_fn=default_data_collator, pin_memory=True, num_workers=8
    #     )
    
    ''' Model '''

    # TODO: verify this
    # NOTE: this is inherit from sparsegpt, but I don't know the reason
    # disable_torch_init()

    with StateStdout(logger=logger, begin="Loading pretrained model.."):
        model_config = BloomConfig.from_pretrained(args.model_name_or_path, cache_dir=args.model_cache_dir)
        if args.gradient_checkpointing_enable:
            model_config.use_cache = False

        model = BloomForCausalLM.from_pretrained(
            args.model_name_or_path,
            config=model_config,
            torch_dtype="auto",
            cache_dir=args.model_cache_dir
        )
        if accelerator.mixed_precision == "fp16":
            model.half()
        elif accelerator.mixed_precision == "bf16":
            model.bfloat16()
        else:
            model.float()

        if args.gradient_checkpointing_enable:
            model.gradient_checkpointing_enable()
    
    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Enable lora
    # if args.lora:
    #     lora_modules = ["query_key_value", "self_attention.dense", "dense_h_to_4h", "dense_4h_to_h"]
    #     lora_config = LoraConfig(
    #         task_type=TaskType.CAUSAL_LM,
    #         r=args.lora_rank, target_modules=lora_modules, 
    #         lora_alpha=args.lora_rank, lora_dropout=0., enable_lora=[True]
    #     )
    #     model = get_peft_model(model, lora_config)
    #     model.print_trainable_parameters()
    
    logger.info(f"Model dtype: {model.dtype}\n")
    logger.info(f"\nModel structure:\n{model}\n")

    # Data parallel
    eval_dataloader = accelerator.prepare_data_loader(eval_dataloader)
    # if not args.eval_only:
    #     # For consistent with deepspeed:
    #     # when intergrate with deepspeed, you must feed dataloader to acceleraotr.prepare()
    #     raw_train_dataloader = train_dataloader
    #     train_dataloader = accelerator.prepare_data_loader(train_dataloader)

    # Dense model evaluation
    if args.eval_dense:
        with StateStdout(logger=logger, begin="Dense model eval .."):
            # args.dense_metric = sequential_eval(model, val_data, eval_dataloader, accelerator)
            # TODO: use train data to do eval, will be removed future
            dataloader = DataLoader(
                train_data, batch_size=args.per_device_train_batch_size,
                collate_fn=default_data_collator, pin_memory=True, num_workers=8
            )
            dataloader = accelerator.prepare_data_loader(dataloader)
            args.dense_metric = sequential_eval(model, args.num_layers_aggregated, train_data, dataloader, accelerator)
            
            del dataloader
            gc.collect()

    if args.dense_metric is not None:
        logger.info(f"Dense perplexity: {args.dense_metric}")

    # Log global messages
    per_device_batch_size = args.per_device_train_batch_size \
        if not args.eval_only else args.per_device_eval_batch_size
    total_batch_size = per_device_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_examples = len(train_data) if not args.eval_only else len(val_data)
    # if not args.eval_only:
    #     num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    #     total_steps = num_update_steps_per_epoch * args.num_train_epochs

    logger.info("\n***** BEGIN *****")
    logger.info(f"  Num examples = {num_examples}")
    logger.info(f"  Instantaneous train batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Instantaneous eval batch size per device = {args.per_device_eval_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    if not args.eval_only:
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        # logger.info(f"  Total optimization steps = {total_steps}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")

    # Eval only
    if args.eval_only:
        with StateStdout(logger=logger, begin="NOTE: Program for Eval only"):
            sequential_eval(model, args.num_layers_aggregated, val_data, eval_dataloader, accelerator)
    # Train(& Prune)
    else:
        # NOTE: We only train BloomBlocks
        model.transformer.word_embeddings.requires_grad_(False)
        model.transformer.word_embeddings_layernorm.requires_grad_(False)
        model.transformer.ln_f.requires_grad_(False)
        model.lm_head.requires_grad_(False)

        dtype = model.dtype
        # Current cuda device
        dev = accelerator.device

        use_cache = model.config.use_cache
        model.config.use_cache = False

        # BloomBlocks
        layers = model.transformer.h
        hidden_size = model.config.hidden_size

        num_sub_modules = int(math.ceil(len(layers) / args.num_layers_aggregated))
        finish_prune = [False] * num_sub_modules

        if not args.num_train_samples_for_dataloader:
            args.num_train_samples_for_dataloader = len(train_data)

        for start_data_index in range(0, len(train_data), args.num_train_samples_for_dataloader):
            with StateStdout(logger=logger, begin="Building sub-dataset.."):
                end_data_index = min(start_data_index + args.num_train_samples_for_dataloader, len(train_data))
                sub_train_data = Subset(train_data, range(start_data_index, end_data_index))
            logger.info(f"Sub dataset indices: [{start_data_index}, {end_data_index})")
            
            raw_train_dataloader = DataLoader(
                sub_train_data, batch_size=args.per_device_train_batch_size,
                collate_fn=default_data_collator, pin_memory=True, num_workers=8
            )
            train_dataloader = accelerator.prepare_data_loader(raw_train_dataloader)

            num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
            total_steps = num_update_steps_per_epoch * args.num_train_epochs

            # Hidden states from previous sparse layer
            hs_batches = []
            # Hidden states from teacher
            teacher_hs_batches = []

            alibi_batches = []
            attention_mask_batches = []

            # Output hidden states from previous dense layer for pruning
            prune_hidden_states = []
            prune_alibis = []
            prune_attention_masks = []

            # Collect outputs from embedding layer, that is, initial batched hidden states.
            with StateStdout(logger=logger, begin="Collecting outputs from embedding layer.."):
                ''' CPU -> GPU '''

                model.transformer.word_embeddings.to(dev)
                model.transformer.word_embeddings_layernorm.to(dev)
                layers[0].to(dev)

                # Collected hidden states
                hs = torch.zeros(
                    (per_device_batch_size, args.max_seq_length, hidden_size),
                    dtype=dtype, device=dev
                )
                # 记录当前 forward 到第几个 BloomBlock 以及对应的一些状态
                cache = {'attention_mask': None, 'alibi': None}
                # 以下会仅经过了：Embedding 层 -> LayerNorm -> the 1st BloomBlock
                # 从而将每个样本在输入第一个 BloomBlock 前的 hidden state 记录下来
                layers[0] = BatchCatcher(layers[0], cache, hs)

                for batch in tqdm(train_dataloader, desc="Collect output embeddings", disable=not accelerator.is_local_main_process):
                    # BloomBlock 前的部分(Embedding 层)不需要梯度
                    with torch.no_grad():
                        try:
                            model(**batch)
                        except ValueError:
                            pass

                    # Collect outputs from embedding layer
                    # Note: remember to put them on cpu in case of running out of gpu memory.
                    hs_batches.append(hs.cpu())
                    teacher_hs_batches.append(hs_batches[-1].clone())
                    alibi_batches.append(cache['alibi'].cpu())
                    attention_mask_batches.append(cache['attention_mask'].cpu())

                # Extract layer from BatchCacher
                layers[0] = layers[0].module

                ''' GPU -> CPU '''

                layers[0].cpu()
                model.transformer.word_embeddings.cpu()
                model.transformer.word_embeddings_layernorm.cpu()

                # We don't need train dataloader anymore,
                # cuz all batched hidden states has been cached. 
                del hs, cache, train_dataloader
                gc.collect()
                torch.cuda.empty_cache()

            # Multi-blocks forwarding, i.e. 
            # it is turn to the next multiple blocks only after the previous multiple blocks done with all batches
            for layer_i in range(0, len(layers), args.num_layers_aggregated):
                layer_j = min(layer_i + args.num_layers_aggregated, len(layers))
                sub_modules = MultiBloomBlocks(*layers[layer_i:layer_j])
                logger.info(f"sub-modules dtype: {next(iter(sub_modules.parameters())).dtype}")

                msg = "Collecting output hidden states from teacher.."
                with StateStdout(logger=logger, begin=msg):
                    sub_modules.eval()
                    sub_modules.to(dev)

                    with torch.no_grad():                    
                        for batch_i, batch in tqdm(
                            enumerate(teacher_hs_batches), desc="Collect teacher outputs",
                            disable=not accelerator.is_local_main_process
                        ):
                            ''' CPU -> GPU '''

                            # Output hidden states from previous dense layer
                            teacher_hs = teacher_hs_batches[batch_i].to(dev)
                            alibi = alibi_batches[batch_i].to(dev)
                            attention_mask = attention_mask_batches[batch_i].to(dev)

                            # [BE CAREFUL] detach and thus cut out the relationship between layers
                            teacher_hs_batches[batch_i] = sub_modules(teacher_hs, attention_mask=attention_mask, alibi=alibi)[0].cpu()

                            ''' Release memories '''

                            del teacher_hs, alibi, attention_mask
                            gc.collect()
                            torch.cuda.empty_cache()

                    sub_modules.cpu()
                    sub_modules.train()

                # Optimizer
                if args.optimizer_type == 'adamw':
                    optimizer = torch.optim.AdamW(sub_modules.parameters(), lr=args.learning_rate)
                elif args.optimizer_type == 'sgd':
                    optimizer = torch.optim.SGD(sub_modules.parameters(), args.learning_rate)
                else:
                    raise NotImplementedError(f"Got not supported optimizer: {args.optimizer_type}.")

                # Lr scheduler
                num_update_steps_total = math.ceil(len(raw_train_dataloader) / args.gradient_accumulation_steps) * args.num_train_epochs
                lr_scheduler = get_scheduler(
                    name=args.lr_scheduler_type,
                    optimizer=optimizer,
                    num_warmup_steps=args.num_warmup_steps * accelerator.gradient_accumulation_steps,
                    num_training_steps=num_update_steps_total * accelerator.gradient_accumulation_steps
                )
                
                # For consistent with deepspeed, must put them together to 'prepare()'
                sub_modules, train_dataloader, optimizer, lr_scheduler = accelerator.prepare(
                    sub_modules, raw_train_dataloader,
                    optimizer, lr_scheduler
                )
                unwrapped_modules = accelerator.unwrap_model(sub_modules)

                # NOTE: here, step means number of epochs
                completed_step = 0
                completed_prune_times = 0
                # Only show the progress bar once on each machine.
                progress_bar = tqdm(range(total_steps), desc=f"Training layer {layer_i + 1}~{layer_j}", disable=not accelerator.is_local_main_process)

                sub_module_index = layer_i // args.num_layers_aggregated

                for epoch in range(args.num_train_epochs):
                    # Pruning
                    if args.sparse and not finish_prune[sub_module_index] and completed_prune_times < len(args.sparsities) \
                        and (args.min_layer <= layer_i < args.max_layer) and \
                            completed_step == args.sparse_steps[completed_prune_times]:
                        sparsity = args.sparsities[completed_prune_times]
                        
                        ''' Random select pruning samples(hidden states) from training samples(hidden states) '''

                        num_prune_batches = min(args.num_prune_samples // hs_batches[0].size(0), len(hs_batches))
                        prune_batch_indices = random.sample(range(len(hs_batches)), num_prune_batches)
                        
                        # NOTE: .clone(): make a copy of training data to prevent influences
                        # (num_prune_batches * train_batch_size, seq_length, hidden_size)
                        prune_hidden_states = torch.cat([hs_batches[batch_i] for batch_i in prune_batch_indices])
                        prune_alibis = torch.cat([alibi_batches[batch_i] for batch_i in prune_batch_indices])
                        prune_attention_masks = torch.cat([attention_mask_batches[batch_i] for batch_i in prune_batch_indices])
                        logger.info(f"{prune_hidden_states.size(0)} pruning samples selected.")

                        # Per layer pruning
                        if args.per_layer_pruning:
                            sub_modules.cpu()
                            for block_i, block in enumerate(unwrapped_modules):
                                with StateStdout(logger=logger, begin=f"Pruning layer {layer_i + block_i + 1} to sparsity {sparsity} .."):
                                    layer_prune(
                                        block, sparsity,
                                        prune_hidden_states, prune_alibis, prune_attention_masks,
                                        accelerator, pruner=args.pruner, prune_target=args.prune_only,
                                        percdamp=args.percdamp, prefix=f"L{layer_i + block_i + 1}"
                                    )
                            sub_modules.to(dev)
                        # Multi layers pruning
                        else:
                            with StateStdout(logger=logger, begin=f"Pruning layer {layer_i + 1}-{layer_j} to sparsity {sparsity} .."):
                                layer_prune(
                                    unwrapped_modules, sparsity,
                                    prune_hidden_states, prune_alibis, prune_attention_masks,
                                    accelerator, pruner=args.pruner, prune_target=args.prune_only,
                                    percdamp=args.percdamp, prefix=f"L{layer_i + 1}_{layer_j}"
                                )
                        
                        completed_prune_times += 1
                        if completed_prune_times == len(args.sparsities):
                            finish_prune[sub_module_index] = True

                        del prune_hidden_states, prune_alibis, prune_attention_masks
                        gc.collect()
                        torch.cuda.empty_cache()

                    sub_modules.train()
                    # NOTE: Set initial gradients to None can save some memories
                    optimizer.zero_grad(set_to_none=True)

                    # Loss among all data
                    modules_loss = 0.
                    # Cosine similarity(vs teacher) among all data
                    modules_similarity = 0.

                    # Train loop
                    for batch_i, (hs, teacher_out_hs, alibi, attention_mask) in \
                        enumerate(zip(hs_batches, teacher_hs_batches, alibi_batches, attention_mask_batches)):
                        # Perform gradient accumulation automatically
                        with accelerator.accumulate(sub_modules):
                            ''' CPU -> GPU '''

                            hs = hs.to(dev)
                            teacher_out = teacher_out_hs.to(dev)

                            alibi = alibi.to(dev)
                            attention_mask = attention_mask.to(dev)

                            # Forward
                            out = sub_modules(hs, attention_mask=attention_mask, alibi=alibi)[0]

                            ''' Align with teacher '''

                            # Norm before computing loss, this is for stability
                            var = out.pow(2).mean(dim=-1, keepdim=True)
                            teacher_var = teacher_out.pow(2).mean(dim=-1, keepdim=True)
                            loss = F.mse_loss(
                                out * torch.rsqrt(var + 1e-8),
                                teacher_out * torch.rsqrt(teacher_var + 1e-8)
                            )
                            del var, teacher_var

                            # If mixed precision training, accelerator will scale loss here
                            accelerator.backward(loss)
                            loss = loss.item()
                            modules_loss += loss

                            ''' Cosine similarity between student & teacher '''

                            # No need for gradients
                            with torch.no_grad():
                                sim = F.cosine_similarity(
                                    out.reshape(-1, hidden_size),
                                    teacher_out.reshape(-1, hidden_size)
                                ).mean()
                            del out, teacher_out

                            # Reduce across all process
                            sim = accelerator.reduce(sim, reduction="mean").item()
                            modules_similarity += sim

                            # Update parameters
                            optimizer.step()
                            # Update lr
                            lr_scheduler.step()

                            optimizer.zero_grad(set_to_none=True)

                        # Log training info
                        if batch_i % 100 == 0 or batch_i == len(hs_batches) - 1:
                            logger.info(
                                f"epoch {epoch + 1}/{args.num_train_epochs}\t"
                                f"step {batch_i + 1}/{len(hs_batches)}\t"
                                f"lr {optimizer.param_groups[0]['lr']}\t"
                                f"loss {loss}\tsimilarity {sim}\t"
                                f"mean loss {modules_loss / (batch_i + 1)}\t"
                                f"mean similarity {modules_similarity / (batch_i + 1)}\n"
                            )

                        # Checks if the accelerator has performed an optimization step behind the scenes
                        if accelerator.sync_gradients:
                            completed_step += 1
                            progress_bar.update()
                            
                            # Apply sparse mask
                            if MASK:
                                with torch.no_grad():
                                    if args.per_layer_pruning:
                                        for block_i, block in enumerate(unwrapped_modules):
                                            subset = find_layers(block, name=f"L{layer_i + block_i + 1}")
                                            for name, module in subset.items():
                                                mask = MASK[name].to(module.weight.device)
                                                module.weight.data = module.weight.data * mask
                                    else:
                                        subset = find_layers(unwrapped_modules, name=f"L{layer_i + 1}_{layer_j}")
                                        for name, module in subset.items():
                                            mask = MASK[name].to(module.weight.device)
                                            module.weight.data = module.weight.data * mask

                                del subset, mask
                                gc.collect()
                                torch.cuda.empty_cache()

                    modules_loss /= len(hs_batches)
                    modules_similarity /= len(hs_batches)
                    logger.info(
                        f"[Layer {layer_i + 1}~{layer_j}]\t Epoch {epoch + 1}\t"
                        f"Loss {modules_loss}\t Similarity {modules_similarity}\n"
                    )
                    
                    ''' Release memories '''

                    gc.collect()
                    torch.cuda.empty_cache()

                    # Done if it is close enough to teacher
                    if modules_similarity > 0.998:
                        logger.info(f"NOTE: Similarity > 0.998, layer {layer_i + 1}~{layer_j} are DONE!\n")

                        remain_steps = (args.num_train_epochs - epoch - 1) * num_update_steps_per_epoch
                        completed_step += remain_steps
                        progress_bar.update(n=remain_steps)

                        break

                # Clear sparse mask, cuz it is no use for next layer if we only train 1 loop with dataloader.
                if args.num_train_samples_for_dataloader == len(train_data):
                    MASK.clear()
                
                # Collect hidden states for next layer.
                if layer_j < len(layers):
                    with StateStdout(logger=logger, begin=f"Collecting input hidden states for next multi-layers.."):
                        with torch.no_grad():
                            for batch_i, (hs, alibi, attention_mask) in tqdm(
                                enumerate(
                                    zip(hs_batches, alibi_batches, attention_mask_batches)
                                ),
                                disable=not accelerator.is_local_main_process
                            ):
                                hs = hs.to(dev)
                                alibi = alibi.to(dev)
                                attention_mask = attention_mask.to(dev)
                                hs_batches[batch_i] = sub_modules(hs, attention_mask=attention_mask, alibi=alibi)[0].detach().cpu()

                        del hs, alibi, attention_mask
                        gc.collect()
                        torch.cuda.empty_cache()

                # GPU -> CPU
                for layer_ind in range(layer_i, layer_j):
                    layers[layer_ind].cpu()
                del sub_modules, unwrapped_modules, optimizer, lr_scheduler, train_dataloader

                # Do eval for finetuned sparse model if needed.
                if args.eval_finetuned_sparse:
                    with StateStdout(logger=logger, begin=f"Eval finetuned layer {layer_i + 1}~{layer_j}.."):
                        # TODO: use train data to do eval, future will use val data instead
                        dataloader = DataLoader(
                            train_data, batch_size=args.per_device_train_batch_size * 2,
                            collate_fn=default_data_collator, pin_memory=True, num_workers=8
                        )
                        # Ditributed setup
                        dataloader = accelerator.prepare_data_loader(dataloader)
                        sequential_eval(model, args.num_layers_aggregated, train_data, dataloader, accelerator)

                        del dataloader
                
                # Releases all references to the internal objects stored and call the garbage collector
                accelerator.clear()

        # Clear sparse mask.
        MASK.clear()

        model.config.use_cache = use_cache
        del hs_batches, teacher_hs_batches, alibi_batches, attention_mask_batches

        logger.info("Training finshed!")
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
    logger.info("System End.")
