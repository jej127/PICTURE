#!/bin/sh

for i in {1..5}
    do
        python evaluate_bigtom.py \
            --eval_model gpt-3.5-turbo \
            --method ours \
            --all
    done
