#!/bin/bash

python -m llava.eval.model_vqa_loader \
    --model-path checkpoints/llava-v1.6-7b \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/llava-v1.6-7b.jsonl \
    --temperature 0 \
    --square_eval True \
    --conv-mode vicuna_v1

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment llava-v1.6-7b

cd eval_tool

python calculation.py --results_dir answers/llava-v1.6-7b
