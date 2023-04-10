#!/bin/bash

SPARSITY=${1}
LOG=${2}

nohup python -u bloom.py bigscience/bloom-560m wikipedia \
    --sparsity $SPARSITY \
    --eval_dense \
    >> $LOG.log 2>&1 &
