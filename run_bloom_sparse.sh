#!/bin/bash

LOG=${1}

CUDA_VISIBLE_DEVICES=0 \
nohup accelerate launch run_bloom_sequential_clm.py \
    --model_name bigscience/bloom \
    --model_name_or_path /ssd6/xiangyangkan/models/huggingface-models/bloom-176b-32x \
    --model_cache_dir /ssd6/xiangyangkan/models/huggingface-models/bloom-176b-32x \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --data_cache_dir /ssd1/datasets/wikitext \
    --max_seq_length 256 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --learning_rate 1e-5 \
    --num_prune_samples 1280 \
    --per_device_prune_batch_size 1280 \
    --sparsities 0.96875 \
    --sparse_steps 0 \
    --pruner abc_solver \
    --percdamp 0.1 \
    --path_to_dense /ssd6/xiangyangkan/models/huggingface-models/bloom-176b \
    --output_dir bloom-176b-32x-accelerate \
    --debug \
    >> $LOG.log 2>&1 &
