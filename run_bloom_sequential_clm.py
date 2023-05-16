import os
import math
import random
import logging
import argparse

import torch
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

from peft import LoraConfig, TaskType, get_peft_model

from sequential import SequentialForward, SparseBloomSequential
# from sequential_single_gpu_fp32 import SequentialForward, SparseBloomSequential
# from sequential_single_gpu_amp import SequentialForward, SparseBloomSequential


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


def disable_torch_init():

    def skip(*args, **kwargs):
        pass

    torch.nn.init.normal_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.kaiming_uniform_ = skip


def sequential_eval(seq_model: SequentialForward, dataset, dataloader, accelerator, verbose=True, **kwargs):
    losses = []
    for batch in tqdm(dataloader, disable=not accelerator.is_local_main_process):
        loss = seq_model.forward(batch, train=False, **kwargs)
        batch_size = batch['input_ids'].size(0)
        # (num_devices*batch_size,)
        losses.append(accelerator.gather(loss.repeat(batch_size)).cpu())

    losses = torch.cat(losses)
    losses = losses[:len(dataset)]

    try:
        perplexity = math.exp(torch.mean(losses))
    except OverflowError:
        perplexity = float("inf")
    
    del losses
    if verbose:
        logger.info(f"Perplexity: {perplexity}\n")

    return perplexity


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
        "--model_name",
        type=str,
        required=True,
        help="Model identifier from huggingface.co/models."
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
        "--use_gradient_checkpointing",
        action="store_true",
        help="Whether to use gradient checkpointing, this is for memory efficiency."
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
        "--pruner",
        type=str,
        default="sparsegpt",
        help="Pruner type."
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

    if args.debug:
        # Only 16 samples will be used to debug
        args.num_prune_samples = 16
        args.per_device_prune_batch_size = 16
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
    
    # Data split
    val_data = data['validation']
    # TODO: uncomment below
    # if not args.eval_full_data and len(val_data) > 1280:
    #     data['validation'] = data['validation'].select(range(1280))
    #     logger.info("NOTE: only 1280 samples of validation data selected.")
    # if not args.eval_only:
    #     train_data = data['train']
    # TODO: remove future, just use test set for better performance currently
    train_data = data['test']
    val_data = data['test']
    if args.debug:
        train_data = train_data.select(range(96))
        # NOTE: debug also means overfit
        val_data = train_data
        logger.info(f"NOTE: debug mode on! only 96 train(eval) data samples selected.")

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
    if not args.eval_only:
        train_dataloader = DataLoader(
            train_data, batch_size=args.per_device_train_batch_size, shuffle=True,
            collate_fn=default_data_collator, pin_memory=True, num_workers=8
        )
    
    # Model
    # NOTE: this is inherit from sparsegpt, but I don't know the reason
    disable_torch_init()

    model_config = BloomConfig.from_pretrained(args.model_name_or_path, cache_dir=args.model_cache_dir)
    model = BloomForCausalLM.from_pretrained(
        args.model_name_or_path,
        config=model_config,
        # torch_dtype=torch.bfloat16,
        cache_dir=args.model_cache_dir
    )

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Enable gradient checkpointing for memory efficiency
    if args.use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
    # Enable lora
    if args.lora:
        lora_modules = ["query_key_value", "self_attention.dense", "dense_h_to_4h", "dense_4h_to_h"]
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_rank, target_modules=lora_modules, 
            lora_alpha=args.lora_rank, lora_dropout=0., enable_lora=[True]
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    logger.info(f"Model dtype: {model.dtype}\n")
    logger.info(f"\nModel structure:\n{model}\n")

    # Model that performs sequential forwarding
    sequential_model = SparseBloomSequential(model, accelerator, logger=logger, dense_dir=args.path_to_dense)

    # Data parallel
    eval_dataloader = accelerator.prepare(eval_dataloader)
    if not args.eval_only:
        train_dataloader = accelerator.prepare(train_dataloader)

    # Dense model evaluation
    if args.eval_dense:
        with StateStdout(logger=logger, begin="Dense model eval"):
            # TODO: verify this
            if args.lora:
                with model.disable_adapter():
                    sequential_eval(sequential_model, val_data, eval_dataloader, accelerator)
            else:
                sequential_eval(sequential_model, val_data, eval_dataloader, accelerator)

    # Log global messages
    per_device_batch_size = args.per_device_train_batch_size \
        if not args.eval_only else args.per_device_eval_batch_size
    total_batch_size = per_device_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_examples = len(train_data) if not args.eval_only else len(val_data)
    if not args.eval_only:
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        total_steps = num_update_steps_per_epoch * args.num_train_epochs

    logger.info("\n***** BEGIN *****")
    logger.info(f"  Num examples = {num_examples}")
    logger.info(f"  Instantaneous batch size per device = {per_device_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    if not args.eval_only:
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Total optimization steps = {total_steps}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info("\n")

    # Eval
    if args.eval_only:
        with StateStdout(logger=logger, begin="Eval only"):
            sequential_eval(sequential_model, val_data, eval_dataloader, accelerator)
    # Train
    else:
        if args.dump_dense:
            os.makedirs(args.path_to_dense, exist_ok=True)
            with StateStdout(logger=logger, begin=f"Dumping dense to {args.path_to_dense}"):
                for i, layer in tqdm(
                    enumerate(sequential_model.model.transformer.h),
                    disable=not accelerator.is_local_main_process
                ):
                    torch.save(layer.state_dict(), os.path.join(args.path_to_dense, f"layer_{i}"))
        
        completed_step = 0
        completed_prune_times = 0

        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(total_steps), disable=not accelerator.is_local_main_process)

        def lr_schedule():
            if completed_step < args.num_warmup_steps:
                return float(completed_step) / float(max(1, args.num_warmup_steps))
            return max(0.0, float(total_steps - completed_step)) / float(max(1, total_steps - args.num_warmup_steps))

        for epoch in range(args.num_train_epochs):
            # Loss yeild by each layer
            layer_losses = [0.] * len(sequential_model.model.transformer.h)
            # Cosine similarity between each layer's output hidden states of sparse & dense
            layer_similarities = [0.] * len(sequential_model.model.transformer.h)

            for step, batch in enumerate(train_dataloader):
                # Pruning when meeting a specified step
                if args.sparse and completed_prune_times < len(args.sparsities) \
                    and completed_step == args.sparse_steps[completed_prune_times]:
                    target_indices = random.sample(range(len(train_data)), args.num_prune_samples)
                    prune_data = train_data.select(target_indices)
                    # TODO: accelerate.prepare()
                    prune_dataloader = DataLoader(
                        prune_data, batch_size=args.per_device_prune_batch_size,
                        collate_fn=default_data_collator, pin_memory=True, num_workers=8
                    )

                    logger.info(f"Num prune samples total = {args.num_prune_samples}")
                    logger.info(f"Prune batch size per device = {args.per_device_prune_batch_size}")

                    sparsity = args.sparsities[completed_prune_times]
                    with StateStdout(logger=logger, begin=f"Pruning to sparsity {sparsity}"):
                        for prune_batch in tqdm(prune_dataloader, desc="Pruning", disable=not accelerator.is_local_main_process):
                            sequential_model.forward(
                                prune_batch, prune=True,
                                min_layer=args.min_layer,
                                max_layer=args.max_layer,
                                prune_only=args.prune_only,
                                sparsity=sparsity,
                                prunen=args.prunen,
                                prunem=args.prunem,
                                pruner=args.pruner,
                                percdamp=args.percdamp
                            )

                    completed_prune_times += 1
                    del prune_data, prune_dataloader

                    # Eval after pruning
                    with StateStdout(logger=logger, begin=f"Eval after pruning"):
                        sequential_eval(
                            sequential_model,
                            val_data, eval_dataloader, accelerator,
                            hard_sparse_weight=args.gmp, sparse_ratio=sparsity
                        )

                # Schedule lr
                lr = args.learning_rate * lr_schedule()
                # Loss yeild by each layer forwarding on this batch
                layer_loss_batch, layer_similarity_batch = sequential_model.forward(batch, lr=lr)
                for i, (loss, sim) in enumerate(zip(layer_loss_batch, layer_similarity_batch)):
                    layer_losses[i] += loss / len(train_dataloader)
                    layer_similarities[i] += sim / len(train_dataloader)
                
                # Show loss when a specified time arrival
                if step % (10 * args.gradient_accumulation_steps) == 0:
                    for i, (loss, sim) in enumerate(zip(layer_loss_batch, layer_similarity_batch)):
                        logger.info(f"layer {i}\t loss {loss}\t similarity {sim}\n")

                if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    completed_step += 1
                    progress_bar.update()
                
                # Eval when meeting a specified step
                # if (step + 1) % (100 * args.gradient_accumulation_steps) == 0:
                #     with StateStdout(logger=logger, begin=f"Eval in step{step + 1}"):
                #         ppl = sequential_eval(
                #             sequential_model,
                #             val_data, eval_dataloader, accelerator, verbose=False,
                #             hard_sparse_weight=args.gmp, sparse_ratio=(sparsity if completed_prune_times else 0.)
                #         )
                #         logger.info(f"Epoch {epoch + 1}\t Step {step + 1}\t Perplexity {ppl}\n")

            # Show epoch loss of each layer
            logger.info(f"[Epoch{epoch + 1} Losses]")
            for i, loss in enumerate(layer_losses):
                logger.info(f"layer {i}\t loss {loss}")
            logger.info(f"\n[Epoch {epoch + 1} Similarities]")
            for j, sim in enumerate(layer_similarities):
                logger.info(f"layer {i}\t similarity {sim}")
            logger.info("\n")

            # Eval when a epoch done
            with StateStdout(logger=logger, begin=f"Eval in epoch{epoch + 1}"):
                ppl = sequential_eval(
                    sequential_model,
                    val_data, eval_dataloader, accelerator, verbose=False,
                    hard_sparse_weight=args.gmp, sparse_ratio=(sparsity if completed_prune_times else 0.)
                )
                logger.info(f"Epoch {epoch + 1}\t Perplexity {ppl}\n")
        
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
