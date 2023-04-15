#!/bin/bash

LOG=${1}

CUDA_VISIBLE_DEVICES=0 \
nohup accelerate launch run_bloom_blockwise_clm.py \
    --model_name_or_path bigscience/bloom-7b1 \
    --model_cache_dir /ssd1/models/bloom \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --data_cache_dir /ssd1/datasets/wikitext \
    --num_train_epochs 100 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --learning_rate 1 \
    --optimizer_type sgd \
    --lr_scheduler_type constant \
    --num_prune_samples 128 \
    --per_device_prune_batch_size 128 \
    --eval_dense \
    --sparse \
    --sparsities 0.5 \
    --sparse_steps 0 \
    --path_to_dense bloom-7b1-dense \
    --output_dir bloom-7b1-2x-blockwise-sgd-lr1_constant \
    >> $LOG.log 2>&1 &
