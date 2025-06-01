#!/bin/sh

for i in {1..5}
    do
        python evaluate_tomi.py \
            --eval_model gpt-3.5-turbo \
            --category all \
            --method ours \
            --temperature 0.0
    done