#!/bin/bash

CKPT="llava-v1.5-13b-crop-v3-pro"

python -m ROI.llava.eval.model_vqa_loader \
    --swi-model-path checkpoints/switch-llava-7b \
    --region-model-path checkpoints/region-llava-7b \
    --seg-model-path checkpoints/seg-llava-7b \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/$CKPT.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment $CKPT
cd eval_tool

python calculation.py --results_dir answers/$CKPT
