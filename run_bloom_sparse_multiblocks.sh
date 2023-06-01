#!/bin/bash

LOG=${1}

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 \
OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
nohup accelerate launch run_bloom_multiblocks_clm.py \
    --model_name_or_path bigscience/bloom-7b1 \
    --model_cache_dir /ssd1/models/bloom \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --data_cache_dir /ssd1/datasets/wikitext \
    --max_seq_length 512 \
    --num_train_epochs 1 \
    --max_num_train_samples 960 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 5 \
    --learning_rate 2e-4 \
    --optimizer_type adamw \
    --lr_scheduler_type linear \
    --sparse \
    --sparsities 0.5 \
    --sparse_steps 0 \
    --pruner abc_solver \
    --percdamp 0.1 \
    --num_prune_samples 160 \
    --num_layers_aggregated 3 \
    --eval_dense \
    --eval_finetuned_sparse \
    --mixed_precision bf16 \
    --output_dir bloom-7b1-2x-3blocks \
    >> $LOG.log 2>&1 &
