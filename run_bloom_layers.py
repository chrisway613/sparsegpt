import os
import gc
import math
import random
import logging
import argparse
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers

from copy import deepcopy
from tqdm.auto import tqdm
from datetime import timedelta
from typing import Optional, Tuple

from torch.utils.data import DataLoader

from accelerate.utils import set_seed
from accelerate.logging import get_logger
from accelerate import Accelerator, InitProcessGroupKwargs

from transformers import default_data_collator, SchedulerType, get_scheduler
from transformers.models.bloom import BloomConfig, BloomModel, BloomPreTrainedModel, BloomForCausalLM, BloomTokenizerFast

from chatllm_data_utils import create_prompt_dataset

from modelutils import find_layers
from sparsegpt import SparseGPT, ABCSolver


logger = get_logger(__name__)

MASK = {}


class StateStdout:
    """ A context manager for print out some messages of current state."""

    def __init__(self, logger=None, begin="Start", end="Done!"):
        self.logger = logger or get_logger("stdout")

        self.begin = begin
        self.end = end

    def __enter__(self):
        self.logger.info(f"{self.begin}")
        return self

    def __exit__(self, *exc):
        self.logger.info(f"{self.end}")


class BloomLayers(BloomPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"h.*.self_attention.scale_mask_softmax.causal_mask"]
                                       
    def __init__(self, config: BloomConfig, pretrained_bloom: BloomPreTrainedModel = None) -> None:
        super().__init__(config)
        self.transformer = BloomModel(config)
        # Initialize weights and apply final processing
        self.post_init()

        if pretrained_bloom is not None:
            self.init_from_pretrained_model(pretrained_bloom)
        
        self.transformer.word_embeddings.requires_grad_(False)
        self.transformer.word_embeddings_layernorm.requires_grad_(False)
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **deprecated_arguments
    ):
        if deprecated_arguments.pop("position_ids", False) is not False:
            # `position_ids` could have been `torch.Tensor` or `None` so defaulting pop to `False` allows to detect if users were passing explicitly `None`
            warnings.warn(
                "`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore"
                " passing `position_ids`.",
                FutureWarning,
            )
        if len(deprecated_arguments) > 0:
            raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")
        
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        
        return transformer_outputs

    def init_from_pretrained_model(self, bloom_model: BloomPreTrainedModel):
        with StateStdout(logger=logger, begin="Initializing from pretrained model.."):
            self.transformer.word_embeddings.load_state_dict(bloom_model.transformer.word_embeddings.state_dict())
            self.transformer.word_embeddings_layernorm.load_state_dict(bloom_model.transformer.word_embeddings_layernorm.state_dict())
            
            for layer, pretrained_layer in zip(self.transformer.h, bloom_model.transformer.h):
                layer.load_state_dict(pretrained_layer.state_dict())


def kd_criterion(outputs, teacher_outputs):
    all_hs = outputs.hidden_states
    teacher_all_hs = teacher_outputs.hidden_states

    loss, eps = 0., 1e-8
    for hs, teacher_hs in zip(all_hs, teacher_all_hs):
        # Layer norm
        var = hs.pow(2).mean(-1, keepdim=True)
        teacher_var = teacher_hs.pow(2).mean(-1, keepdim=True)

        hs = hs * torch.rsqrt(var + eps)
        teacher_hs = teacher_hs * torch.rsqrt(teacher_var + eps)

        loss = loss + F.mse_loss(hs, teacher_hs)
    
    return loss


@torch.no_grad()
def cos_similarity(outputs, teacher_outputs):
    hidden_size = outputs.size(-1)

    sim = F.cosine_similarity(
        outputs.reshape(-1, hidden_size).float(),
        teacher_outputs.reshape(-1, hidden_size).float()
    ).mean()
    torch.cuda.empty_cache()

    # Reduce across all process
    return accelerator.reduce(sim, reduction="mean").item()


@torch.no_grad()
def sequential_pruning(
    model: torch.nn.Module, dataloader: DataLoader,
    accelerator: Accelerator, num_samples: int, seq_length: int,
    sparsity: float, percdamp: float = .01, pruner_type: str = "sparsegpt",
    sparse_thresh: float = 1e-10, dtype=torch.float, output_dir=None
):
    mode = model.training
    model.eval()

    use_cache = model.config.use_cache
    model.config.use_cache = False

    device = accelerator.device

    model.transformer.word_embeddings.to(device)
    model.transformer.word_embeddings_layernorm.to(device)

    layers = model.transformer.h    
    layers[0].to(device)

    batch_size = next(iter(dataloader))["input_ids"].size(0)
    num_batches = num_samples // batch_size

    cache = {'i': 0, 'attention_mask': None, 'alibi': None}
    hs = torch.zeros((num_samples, seq_length, model.config.hidden_size), dtype=dtype)

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            # hs[cache['i']] = inp

            # cache['i'] += 1

            hs[cache['i']:(cache['i'] + batch_size)] = inp

            cache['i'] += batch_size
            cache['attention_mask'] = kwargs['attention_mask']
            cache['alibi'] = kwargs['alibi']

            raise ValueError
        
    layers[0] = Catcher(layers[0])

    for i, batch in tqdm(enumerate(dataloader, start=1), desc="Pre-layers: per batch forwarding", disable=not accelerator.is_local_main_process):
        # for j in range(batch_size):
        #     sample = {k: v[j].unsqueeze(0) for k, v in batch.items()}
        #     try:
        #         model(**sample)
        #     except ValueError:
        #         pass
        
        try:
            model(**batch)
        except ValueError:
            pass
        
        if i == num_batches:
            break

    layers[0] = layers[0].module
    layers[0].cpu()
    model.transformer.word_embeddings_layernorm.cpu()
    model.transformer.word_embeddings.cpu()

    attention_mask = cache.pop('attention_mask')
    alibi = cache.pop('alibi')
    del cache

    torch.cuda.empty_cache()

    for i in range(len(layers)):
        layer = layers[i].to(device)
        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = ABCSolver(subset[name]) if pruner_type == "abc_solver" else SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp
        
        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in tqdm(range(0, num_samples, batch_size), desc="In-layers: per batch forwarding", disable=not accelerator.is_local_main_process):
            # layer(hs[j].unsqueeze(0).to(device), attention_mask=attention_mask, alibi=alibi)
            layer(hs[j:(j + batch_size)].to(device), attention_mask=attention_mask, alibi=alibi)

        for h in handles:
            h.remove()

        with StateStdout(logger=logger, begin=f"Pruning layer{i + 1}.."):
            for name in gpts:
                logger.info(f"Sub-module: {name}")

                if pruner_type == "abc_solver":
                    gpts[name].prune_structured(
                        sparsity,
                        percdamp=percdamp
                    )
                else:
                    gpts[name].fasterprune(
                        sparsity,
                        percdamp=percdamp
                    )

                # Reduce across all processes.
                subset[name].weight.data = accelerator.reduce(subset[name].weight.data, reduction="mean")
                # Binary sparse mask (0|1)
                MASK[name] = (subset[name].weight.abs() > sparse_thresh).to(dtype=dtype, device="cpu")

        for j in range(0, num_samples, batch_size):
            # hs[j] = layer(hs[j].unsqueeze(0).to(device), attention_mask=attention_mask, alibi=alibi)[0]
            hs[j:(j + batch_size)] = layer(hs[j:(j + batch_size)].to(device), attention_mask=attention_mask, alibi=alibi)[0]

        layers[i] = layer.cpu()

        del subset, gpts, layer
        gc.collect()

        torch.cuda.empty_cache()

        if output_dir:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                os.makedirs(output_dir, exist_ok=True)
                dst = os.path.join(output_dir, f"layer{i}.bin")
                with StateStdout(logger=logger, begin=f"Saving sparse weights of layer{i + 1} to {dst}.."):
                    torch.save(layers[i].state_dict(), dst)

    del hs
    gc.collect()

    torch.cuda.empty_cache()

    model.config.use_cache = use_cache
    model.train(mode=mode)


def parse_args():
    parser = argparse.ArgumentParser(description="Distill BLOOM layers.")

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
        "--layers_config_file",
        type=str,
        default=None,
        help="Path to a config file which has configuration for Bloom layers."
    )
    parser.add_argument(
        "--gradient_checkpointing_enable",
        action="store_true",
        help="Activates gradient checkpointing for model."
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
        '--sparsity',
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
        "--sparse_weights_dir",
        type=str,
        default=None,
        help="Path to sparse weights."
    )

    # Global
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

    ''' Global preparation '''

    # For distributed training.
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
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    # Fix the random seed.
    set_seed(args.seed)

    accelerator.wait_for_everyone()

    ''' Model '''

    tokenizer = BloomTokenizerFast.from_pretrained(args.model_name_or_path, cache_dir=args.model_cache_dir)
    if args.max_seq_length > tokenizer.model_max_length:
        args.max_seq_length = tokenizer.model_max_length
        logger.warning(
            f"The max_seq_length passed ({args.max_seq_length}) is larger than the maximum length for the model"
            f"({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}.\n"
        )

    # Load pretrained BLOOM model.
    with StateStdout(logger=logger, begin="Loading pretrained model.."):
        bloom_config = BloomConfig.from_pretrained(args.model_name_or_path, cache_dir=args.model_cache_dir)
        bloom_model = BloomForCausalLM.from_pretrained(
            args.model_name_or_path,
            config=bloom_config,
            cache_dir=args.model_cache_dir,
            torch_dtype="auto"
        )
    
    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = bloom_model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        bloom_model.resize_token_embeddings(len(tokenizer))

    # Building a model only contain a few layers of the BLOOM model.
    with StateStdout(logger=logger, begin="building layers.."):
        model_config = BloomConfig.from_json_file(args.layers_config_file)
        if args.gradient_checkpointing_enable:
            model_config.use_cache = False
        
        model = BloomLayers(model_config, pretrained_bloom=bloom_model)
        if accelerator.mixed_precision == "fp16":
            model.half()
        elif accelerator.mixed_precision == "bf16":
            model.bfloat16()
        else:
            model.float()

        if args.gradient_checkpointing_enable:
            model.gradient_checkpointing_enable()

    logger.info(f"Model dtype: {model.dtype}; Num layers: {len(model.transformer.h)}")
    logger.info(f"Model structure:\n{model}\n")

    # The BLOOM model is no use later
    del bloom_config, bloom_model
    gc.collect()

    teacher = deepcopy(model)
    teacher.eval()

    ''' Data '''
    
    with StateStdout(logger=logger, begin="Loading & processing data.."):
        # Note: The processed data will be saved to a specified directory.
        train_data, val_data = create_prompt_dataset(
            accelerator.local_process_index,
            ["sharegpt:/hdd66/chatllm_datasets/sharegpt-train.json|/hdd66/chatllm_datasets/sharegpt-eval.json"],
            "1,0,0", "./chatllm_data_files", 1, args.seed, tokenizer, args.max_seq_length
        )

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_data)), 3):
        logger.info(f"Sample {index} of the training data: {train_data[index]}.")
    
    # Log the number of samples
    logger.info(f"\tNum training data = {len(train_data)}")
    logger.info(f"\tNum validation data = {len(val_data)}")
    
    # Dataloader
    eval_dataloader = DataLoader(
        val_data, batch_size=args.per_device_eval_batch_size,
        collate_fn=default_data_collator, pin_memory=True, num_workers=8
    )
    train_dataloader = DataLoader(
        train_data, batch_size=args.per_device_train_batch_size, shuffle=True,
        collate_fn=default_data_collator, pin_memory=True, num_workers=8
    )

    # Pruning
    if args.sparse:
        prune_dataloader = accelerator.prepare_data_loader(train_dataloader)
        sequential_pruning(
            model, prune_dataloader, accelerator,
            args.num_prune_samples, args.max_seq_length,
            args.sparsity, percdamp=args.percdamp, pruner_type=args.pruner,
            dtype=model.dtype, output_dir=args.output_dir
        )
        
        del prune_dataloader
        gc.collect()

    # Load sparse weights if existed.
    if args.sparse_weights_dir:
        for i, layer in enumerate(model.transformer.h):
            with StateStdout(logger=logger, begin=f"Loading sparse weights of layer{i + 1}.."):
                path = os.path.join(args.sparse_weights_dir, f"layer{i}.bin")
                sparse_weights = torch.load(path, map_location="cpu")
                layer.load_state_dict(sparse_weights)

                sparse_modules = find_layers(layer)
                for name, module in sparse_modules.items():
                    # Binary sparse mask (0|1)
                    MASK[name] = (module.weight.abs() > 1e-10).to(dtype=model.dtype, device="cpu")
                    sparsity = 1. - MASK[name].mean()
                    logger.info(f"Sparsity of layer{i + 1} sub-module {name}: {sparsity}")

    ''' Optimization '''

    # Optimizer
    if args.optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    elif args.optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate)
    else:
        raise NotImplementedError(f"Got invalid optimizer: {args.optimizer_type}.")

    # Lr scheduler
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs
    )

    ''' Distributed training preparation '''

    model, train_dataloader, eval_dataloader, optimizer, lr_scheduler = accelerator.prepare(
        model,
        train_dataloader, eval_dataloader,
        optimizer, lr_scheduler
    )
    unwrapped_model = accelerator.unwrap_model(model)

    ''' Statistics '''

    total_train_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    total_optimization_steps = num_update_steps_per_epoch * args.num_train_epochs

    logger.info("\n***** BEGIN *****")
    logger.info(f"  Num train examples = {len(train_data)}")
    logger.info(f"  Num val examples = {len(val_data)}")
    logger.info(f"  Instantaneous train batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Instantaneous eval batch size per device = {args.per_device_eval_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Total optimization steps = {total_optimization_steps}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info("\n")

    accelerator.wait_for_everyone()

    ''' Training '''

    progress_bar = tqdm(range(total_optimization_steps), disable=not accelerator.is_local_main_process)

    for epoch in range(args.num_train_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        epoch_loss = 0.
        epoch_similarity = 0.

        for step, batch in enumerate(train_dataloader):
            # Perform gradient accumulation automatically
            with accelerator.accumulate(model):
                # Forward
                outputs = model(**batch, output_hidden_states=True)
                model.cpu()
                # torch.cuda.empty_cache()

                teacher.to(accelerator.device)
                with torch.no_grad():
                    teacher_outputs = teacher(**batch, output_hidden_states=True)
                teacher.cpu()
                # torch.cuda.empty_cache()
                
                # Backward
                loss = kd_criterion(outputs, teacher_outputs)
                model.to(accelerator.device)
                accelerator.backward(loss)

                loss = loss.item()
                epoch_loss += loss

                # Calculate cosine similarity with teacher
                sim = cos_similarity(outputs[0], teacher_outputs[0])
                epoch_similarity += sim

                # Optimize
                optimizer.step()
                lr_scheduler.step()

                optimizer.zero_grad(set_to_none=True)

            # Log info
            if step % 100 == 0 or step == len(train_dataloader) - 1:
                logger.info(
                    f"epoch {epoch + 1}/{args.num_train_epochs}\t"
                    f"step {step + 1}/{len(train_dataloader)}\t"
                    f"lr {optimizer.param_groups[0]['lr']}\t"
                    f"loss {loss}\tsimilarity {sim}\t"
                    f"mean loss {epoch_loss / (step + 1)}\t"
                    f"mean similarity {epoch_similarity / (step + 1)}\n"
                )
            
            if accelerator.sync_gradients:
                progress_bar.update()

                if MASK:
                    with torch.no_grad():
                        for layer in unwrapped_model.transformer.h:
                            target = find_layers(layer)
                            for name, module in target.items():
                                mask = MASK[name].to(device=module.weight.device, dtype=module.weight.dtype)
                                module.weight.data = module.weight.data * mask
                                
                    del target, mask
                    gc.collect()
                    torch.cuda.empty_cache()
        
        epoch_loss /= len(train_dataloader)
        epoch_similarity /= len(train_dataloader)
        logger.info(
            f"Epoch {epoch + 1}/{args.num_train_epochs}\t"
            f"Loss {epoch_loss}\t Similarity {epoch_similarity}\n"
        )

        # Break the training loop if model is qualified.
        if epoch_similarity > 0.998:
            logger.info(f"NOTE: Similarity > 0.998, training DONE!\n")
            break

    accelerator.wait_for_everyone()
    # Releases all references to the internal objects stored and call the garbage collector.
    accelerator.clear()
