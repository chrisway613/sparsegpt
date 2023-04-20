#!/bin/bash

LOG=${1}

CUDA_VISIBLE_DEVICES=0,1 \
nohup accelerate launch dump_hidden_states.py \
    --model_name bigscience/bloom \
    --model_name_or_path /ssd6/xiangyangkan/models/huggingface-models/bloom-176b \
    --model_cache_dir /ssd6/xiangyangkan/models/huggingface-models/bloom-176b \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --data_cache_dir /ssd1/datasets/wikitext \
    --per_device_batch_size 32 \
    --target_layer 64 \
    >> $LOG.log 2>&1 &
