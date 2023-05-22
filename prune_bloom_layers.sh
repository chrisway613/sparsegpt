#!/bin/bash

LOG=${1}

OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
nohup accelerate launch run_bloom_layers.py \
    --model_name_or_path bigscience/bloom-7b1 \
    --model_cache_dir /ssd1/models/bloom \
    --layers_config_file /ssd1/models/bloom/models--bigscience--bloom-7b1/bloom_5layers_config.json \
    --max_seq_length 512 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --learning_rate 5e-5 \
    --optimizer_type adamw \
    --lr_scheduler_type constant \
    --num_prune_samples 160 \
    --sparse \
    --sparsity 0.875 \
    --pruner abc_solver \
    --percdamp 0.1 \
    --output_dir bloom-7b1-5layers-8x-960samples \
    >> $LOG.log 2>&1 &