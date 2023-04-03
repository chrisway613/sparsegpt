import os
import random
import logging
import argparse

import datasets
import transformers

from itertools import chain
from datetime import timedelta

from datasets import load_dataset

from torch.utils.data import DataLoader

from accelerate.utils import set_seed
from accelerate.logging import get_logger
from accelerate import Accelerator, InitProcessGroupKwargs

from transformers import SchedulerType, get_scheduler, default_data_collator
from transformers.models.bloom import BloomConfig, BloomTokenizerFast, BloomForCausalLM


logger = get_logger(__name__)


class StateStdout:
    """ A context manager for print out some messages of current state."""

    def __init__(self, logger=None, begin="Start", end="Done"):
        self.logger = logger or get_logger("stdout")

        self.begin = begin
        self.end = end

    def __enter__(self):
        self.logger.info(f"{self.begin}..")
        return self

    def __exit__(self, *exc):
        self.logger.info(f"{self.end}\n")


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
        "model_cache_dir",
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
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts",
                 "polynomial", "constant", "constant_with_warmup"],
    )

    # Prune
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
        "max_prune_steps",
        type=int,
        default=1000,
        help="Maximum number of pruning steps for a block."
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
        help='Whether to turn on debug mode. If true, only 1000 samples of training data will be selected.'
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
        default="bloom_outputs",
        help="Where to store the results."
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Parse arguments
    args = parse_args()
    logger.info(f"Target sparsity: {args.sparsity}\n")

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
    # Make output directory if not existed
    os.makedirs(args.output_dir, exist_ok=True)

    accelerator.wait_for_everyone()

    # Load dataset
    with StateStdout(logger=logger, begin="Loading datset", end="Done"):
        data = load_dataset(
            args.dataset_name, args.dataset_config_name,
            cache_dir=args.cache_dir, use_auth_token=True
        )
        if 'validation' not in data:
            # val : train = 5% : 95%
            num_val_samples = int(0.05 * len(data['train']))
            val_indices = random.sample(range(len(data['train'])), num_val_samples)
            train_indices = [i for i in range(len(data['train'])) if i not in val_indices]
            assert len(train_indices) + len(val_indices) == len(data['train'])

            data['validation'] = data['train'].select(val_indices)
            data['train'] = data['train'].select(train_indices)
        if not args.eval_full_data:
            data['validation'] = data['validation'].select(range(1000))
            logger.info("NOTE: only 1000 samples of validation data selected.")

    # Tokenize data texts
    tokenizer = BloomTokenizerFast.from_pretrained(args.model_name_or_path, cache_dir=args.model_cache_dir)

    column_names = data['train'].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    with accelerator.main_process_first():
        data = data.map(
            lambda samples: tokenizer(samples[text_column_name]),
            batched=True,
            num_proc=128,
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
            num_proc=128,
            load_from_cache_file=True,
            desc=f"Grouping texts in chunks of {args.max_seq_length}",
        )
    
    train_data, val_data = data['train'], data['validation']
    # TODO: do this when pruning
    # if args.sparse:
    #     target_indices = random.sample(range(len(train_data)), args.num_prune_samples)
    #     prune_data = train_data.select(target_indices)
    #     logger.info(f"NOTE: {args.num_prune_samples} samples of training data selected to be pruning data.\n")
    
    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_data)), 3):
        logger.info(f"Sample {index} of the training data: {train_data[index]}.")
    
    # Dataloader
    train_dataloader = DataLoader(
        train_data, batch_size=args.per_device_train_batch_size, shuffle=True,
        collate_fn=default_data_collator, pin_memory=True, num_workers=8
    )
    eval_dataloader = DataLoader(
        val_data, batch_size=args.per_device_eval_batch_size,
        collate_fn=default_data_collator, pin_memory=True, num_workers=8
    )
    # TODO: do this when pruning
    # if args.sparse:
    #     prune_dataloader = DataLoader(
    #         prune_data, batch_size=args.per_device_prune_batch_size,
    #         collate_fn=default_data_collator, pin_memory=True, num_workers=8
    #     )
    
    # Model
