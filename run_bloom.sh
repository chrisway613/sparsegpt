#!/bin/bash

SPARSITY=${1}  # 0.96875 = 32x; 0.9375 = 16x; 0.875 = 8x
LOG=${2}

nohup python -u bloom.py /ssd6/xiangyangkan/models/huggingface-models/bloom-176b wikitext2 \
    --model_name bigscience/bloom \
    --model_cache_dir /ssd6/xiangyangkan/models/huggingface-models/bloom-176b \
    --data_cache_dir /ssd1/datasets/wikitext \
    --sparsity $SPARSITY \
    --abc_solver \
    --percdamp 0.1 \
    --save /ssd6/xiangyangkan/models/huggingface-models/bloom-176b-8x \
    --cuda_id 0 \
    >> $LOG.log 2>&1 &
