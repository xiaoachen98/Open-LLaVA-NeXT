#!/bin/bash
# srun -p mllm --gres gpu:1 bash scripts/v1_6/eval/mme.sh

CONV_MODE=llava_llama_3
CKPT=$1
CKPT_DIR=${2-'checkpoints'}

python -m llava.eval.model_vqa_loader \
    --model-path ${CKPT_DIR}/${CKPT} \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release \
    --answers-file ./playground/data/eval/MME/answers/${CKPT}.jsonl \
    --temperature 0 \
    --square_eval True \
    --conv-mode $CONV_MODE

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment ${CKPT}

cd eval_tool

python calculation.py --results_dir answers/${CKPT}
