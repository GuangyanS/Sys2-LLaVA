#!/bin/bash

CKPT="llava-v1.5-13b-crop-v3-pro"

python -m ROI.llava.eval.model_vqa_loader \
    --swi-model-path checkpoints/switch-llava-7b \
    --region-model-path checkpoints/region-llava-7b \
    --seg-model-path checkpoints/seg-llava-7b \ 
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./playground/data/eval/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/$CKPT.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/$CKPT.jsonl
