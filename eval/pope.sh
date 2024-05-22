#!/bin/bash

CKPT="llava-v1.5-13b-continue"

python -m ROI.llava.eval.model_vqa_loader \
    --swi-model-path checkpoints/switch-llava-7b \
    --region-model-path checkpoints/region-llava-7b \
    --seg-model-path checkpoints/seg-llava-7b \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder ./playground/data/eval/pope/val2014 \
    --answers-file ./playground/data/eval/pope/answers/$CKPT.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/$CKPT.jsonl
