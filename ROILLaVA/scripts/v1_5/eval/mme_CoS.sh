#!/bin/bash

CKPT="/PATH/TO/CKPT"

python -m llava.eval.model_vqa_loader_CoS \
    --model-path /home/dxleec/gysun/init_weights/CoS-7B-v1-5-lora \
    --question-file /home/dxleec/gysun/datasets/MME/llava_mme.jsonl \
    --image-folder /home/dxleec/gysun/datasets/MME/MME_Benchmark_release_version \
    --answers-file /home/dxleec/gysun/datasets/MME/answers/$CKPT.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

cd /home/dxleec/gysun/datasets/MME

python convert_answer_to_mme.py --experiment $CKPT
cd eval_tool

python calculation.py --results_dir answers/$CKPT
