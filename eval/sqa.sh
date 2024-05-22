#!/bin/bash

CKPT="llava-v1.5-13b-crop-v2"

python -m ROI.llava.eval.model_vqa_science \
    --swi-model-path checkpoints/switch-llava-7b \
    --region-model-path checkpoints/region-llava-7b \
    --seg-model-path checkpoints/seg-llava-7b \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/$CKPT.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/$CKPT.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/$CKPT_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/$CKPT_result.json
