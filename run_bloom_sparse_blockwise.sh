#!/bin/bash

LOG=${1}

CUDA_VISIBLE_DEVICES=5,6 \
nohup accelerate launch run_bloom_blockwise_clm.py \
    --model_name bigscience/bloom \
    --model_name_or_path /ssd6/xiangyangkan/models/huggingface-models/bloom-176b \
    --model_cache_dir /ssd6/xiangyangkan/models/huggingface-models/bloom-176b \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --data_cache_dir /ssd1/datasets/wikitext \
    --num_train_epochs 1000 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --learning_rate 0.1 \
    --optimizer_type sgd \
    --lr_scheduler_type constant \
    --num_prune_samples 128 \
    --per_device_prune_batch_size 128 \
    --dense_metric 15.291 \
    --sparse \
    --sparsities 0.96875 \
    --sparse_steps 0 \
    --pruner abc_solver \
    --percdamp 0.1 \
    --output_dir bloom-176b-32x-blockwise-lr0.1 \
    >> $LOG.log 2>&1 &
