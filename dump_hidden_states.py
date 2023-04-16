import os
import gc
import random
import logging
import argparse

import numpy as np

import torch
import torch.nn as nn

import datasets
import transformers

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

        # Raise an exception error and thus cut the forward chain and catch intermediate resutls
        raise ValueError


def disable_torch_init():
    """ Disable initialization of Pytorch. """

    def skip(*args, **kwargs):
        pass

    torch.nn.init.normal_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.kaiming_uniform_ = skip


@torch.no_grad()
def dump_layer_output_hs(
    model: nn.Module, layer_i: int, dataloader: DataLoader, accelerator: Accelerator, eval: bool = True,
    output_dir: str = None, hard_sparse_weight: bool = False, sparse_ratio: float = 0., verbose: bool = True
):
    """ Dump output hidden states of a target layer. """

    mode = model.training
    use_cache = model.config.use_cache
    model.config.use_cache = False

    model.train(mode=not eval)

    # Embedding layer
    word_embed = model.transformer.word_embeddings
    word_embed_ln = model.transformer.word_embeddings_layernorm
    # All BloomBlocks
    layers = model.transformer.h

    dev = accelerator.device
    dtype = next(iter(model.parameters())).dtype
    hidden_size = model.config.hidden_size

    result = []
    for batch in tqdm(dataloader, desc="Collect batched output hidden states", disable=not accelerator.is_local_main_process):
        ''' CPU -> GPU '''

        # Embedding layer & the 1st Layer Norm
        model.transformer.word_embeddings = word_embed.to(dev)
        model.transformer.word_embeddings_layernorm = word_embed_ln.to(dev)
        # The 1st BloomBlock
        layers[0] = layers[0].to(dev)

        ''' Collect output hidden states from embedding layer '''

        batch_size, seq_length = batch['input_ids'].shape[:2]
        # 记录每个样本在输入 BloomBlock 前的 hidden state
        hs = torch.zeros(
            (batch_size, seq_length, hidden_size),
            dtype=dtype, device=dev
        )
        # 记录当前 forward 到第几个 BloomBlock 以及对应的一些状态
        cache = {'layer_i': 0, 'attention_mask': None, 'alibi': None}

        # 以下会仅经过了：Embedding 层 -> LayerNorm -> the 1st BloomBlock
        # 从而将每个样本在输入第一个 BloomBlock 前的 hidden state 记录下来
        layers[0] = Catcher(layers[0], cache, hs)
        for sample_i in range(batch_size):
            sample = {k: v[sample_i].unsqueeze(0) for k, v in batch.items()}
            try:
                model(**sample)
            except ValueError:
                pass

        ''' GPU -> CPU '''

        layers[0] = layers[0].module.cpu()
        model.transformer.word_embeddings = word_embed.cpu()
        model.transformer.word_embeddings_layernorm = word_embed_ln.cpu()
        accelerator._models.clear()

        # Used for all layers
        alibi = cache.pop('alibi')
        attention_mask = cache.pop('attention_mask')
        del cache

        # Sequential forwarding until the target layer
        for i in range(len(layers[:layer_i + 1])):
            # CPU -> GPU
            # layers[i] = accelerator.prepare(layers[i])
            layers[i] = layers[i].to(dev)

            # 将稀疏的部分置 0
            if hard_sparse_weight:
                subset = find_layers(layers[i])
                for name in subset:
                    W = subset[name].weight.data
                    # 例如稀疏率是75%，那么先有小到大排序，然后将前 75% 的参数值置0
                    thresh = torch.sort(torch.abs(W.flatten()))[0][int(W.numel() * sparse_ratio)]
                    W.data[torch.abs(W.data) <= thresh] = 0

            for sample_i in range(batch_size):
                # Regard output hidden states as the input hidden states of next layer
                hs[sample_i] = layers[i](hs[sample_i].unsqueeze(0), attention_mask=attention_mask, alibi=alibi)[0]
            # The results collected should be put to cpu in case of increasing gpu memory used
            result.append(hs.cpu())

            # GPU -> CPU
            layers[i] = layers[i].cpu()
            accelerator._models.clear()

    del hs

    ''' Save results '''

    # (num_batches * batch_size, seq_length, hidden_size)
    # tensor -> array
    result = torch.cat(result)
    # NOTE: Numpy array do not support bf16
    if result.dtype == torch.bfloat16:
        result = result.half()
        logger.info("NOTE: numpy array do not support bf16, cast to fp16 instead.\n")
    result = result.numpy()
    
    output_dir = output_dir or f"rank{accelerator.process_index}"
    os.makedirs(output_dir, exist_ok=True)

    file = os.path.join(output_dir, f"layer{layer_i}.npz")
    np.savez(file, layer_outputs=result)
    if verbose:
        logger.info(f"NOTE: output hidden states of layer {i} has been saved to {file}.\n")
    
    ''' Release memories '''

    del result
    gc.collect()
    torch.cuda.empty_cache()

    model.config.use_cache = use_cache
    model.train(mode=mode)


def parse_args():
    parser = argparse.ArgumentParser(description="Dump output hidden states of a target BLOOM Layer(BloomBlock) to a specified paht.")

    # Model
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="model identifier from huggingface.co/models, used for initial tokenizer."
    )
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

    # Global
    parser.add_argument(
        "--target_layer",
        type=int,
        required=True,
        help="The target layer of which the output hidden states come from. It should be in range of [0, num_layers - 1]."
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=256,
        help="Max sequence length of a data sample."
    )
    parser.add_argument(
        "--per_device_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the dataloader."
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
        default=None,
        help="Where to store the results."
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    logger.info(f"Target layer: {args.target_layer}\n")

    # Accelerator(for distributed training)
    kwargs_handlers = [InitProcessGroupKwargs(timeout=timedelta(seconds=180000))]
    accelerator = Accelerator(kwargs_handlers=kwargs_handlers)
    
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
    with StateStdout(logger=logger, begin="Loading datset ..", end="Done!"):
        data = load_dataset(
            args.dataset_name, args.dataset_config_name,
            cache_dir=args.data_cache_dir, use_auth_token=True
        )
        # TODO: transfer below to test data
        # if 'validation' not in data:
        #     # val : train = 5% : 95%
        #     num_val_samples = int(0.05 * len(data['train']))
        #     val_indices = random.sample(range(len(data['train'])), num_val_samples)
        #     train_indices = [i for i in range(len(data['train'])) if i not in val_indices]
        #     assert len(train_indices) + len(val_indices) == len(data['train'])

        #     data['validation'] = data['train'].select(val_indices)
        #     data['train'] = data['train'].select(train_indices)

    # Tokenize data texts
    tokenizer = BloomTokenizerFast.from_pretrained(args.model_name, cache_dir=args.model_cache_dir)

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
    
    test_data = data['test']
    logger.info(f"num test data samples with sequence length {args.max_seq_length}: {len(test_data)}.\n")

    # Log a few random samples from the training set:
    for index in random.sample(range(len(test_data)), 3):
        logger.info(f"Sample {index} of the test data: {test_data[index]}.")
    
    # Dataloader
    dataloader = DataLoader(
        test_data, batch_size=args.per_device_batch_size,
        collate_fn=default_data_collator, pin_memory=True, num_workers=8
    )

    # Model
    # NOTE: this is inherit from sparsegpt, but I don't know the reason
    disable_torch_init()

    model_config = BloomConfig.from_pretrained(args.model_name_or_path, cache_dir=args.model_cache_dir)
    model = BloomForCausalLM.from_pretrained(
        args.model_name_or_path,
        config=model_config,
        torch_dtype=torch.bfloat16,  # NOTE: pls feel free to change this. ps: 'auto' = fp16
        cache_dir=args.model_cache_dir
    )
    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
        logger.info(f"NOTE: resize token embeddings to {len(tokenizer)} tokens.\n")
    logger.info(f"\nModel structure:\n{model}\n")
    logger.info(f"Model dtype: {model.dtype}\n")

    # TODO: apply your sparse mask here if needed.

    # Data parallel
    dataloader = accelerator.prepare(dataloader)

    # Dump output hidden states of the target layer
    dump_layer_output_hs(model, args.target_layer, dataloader, accelerator, output_dir=args.output_dir)

    # Releases all references to the internal objects stored and call the garbage collector.
    accelerator.clear()
