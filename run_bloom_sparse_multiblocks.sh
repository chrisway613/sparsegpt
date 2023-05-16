#!/bin/bash

LOG=${1}

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=2,3 \
nohup accelerate launch run_bloom_multiblocks_clm.py \
    --model_name bigscience/bloom-7b1 \
    --model_name_or_path bigscience/bloom-7b1 \
    --model_cache_dir /ssd1/models/bloom \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --data_cache_dir /ssd1/datasets/wikitext \
    --max_seq_length 4096 \
    --num_train_epochs 1000 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --learning_rate 5e-3 \
    --optimizer_type adamw \
    --lr_scheduler_type constant \
    --num_prune_samples 1280 \
    --sparse \
    --sparsities 0.875 \
    --sparse_steps 0 \
    --pruner abc_solver \
    --percdamp 0.1 \
    --num_layers_aggregated 5 \
    --output_dir bloom-7b1-8x-5blocks \
    >> $LOG.log 2>&1 &
