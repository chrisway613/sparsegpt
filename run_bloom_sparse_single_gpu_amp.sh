#!/bin/bash

LOG=${1}

CUDA_VISIBLE_DEVICES=5 \
nohup python -u run_bloom_sequential_clm.py \
    --model_name_or_path bigscience/bloom-7b1 \
    --model_cache_dir /ssd1/models/bloom \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --data_cache_dir /ssd1/datasets/wikitext \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --learning_rate 5e-6 \
    --num_prune_samples 1280 \
    --per_device_prune_batch_size 1280 \
    --eval_dense \
    --sparse \
    --sparsities 0.5 \
    --sparse_steps 0 \
    --path_to_dense bloom-7b1-dense \
    --output_dir bloom-7b1-2x-amp \
    >> $LOG.log 2>&1 &
