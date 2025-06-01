#!/bin/sh

n=1

for i in {1..5}
    do
        CUDA_VISIBLE_DEVICES=$n python evaluate_bigtom.py \
            --eval_model meta-llama/Llama-2-7b-chat-hf \
            --all \
            --method ours \
            --do_sample \
            --temperature 0.3
    done