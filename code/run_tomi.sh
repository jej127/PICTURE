#!/bin/sh

n=1

CUDA_VISIBLE_DEVICES=$n python evaluate_tomi.py \
    --eval_model meta-llama/Llama-2-7b-chat-hf \
    --method ours \
    --category all \
    --temperature 0.3 \
    --do_sample