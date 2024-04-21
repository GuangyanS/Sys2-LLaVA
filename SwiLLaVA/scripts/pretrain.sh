#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/home/gs3260/gysun/projs/Sys2-LLaVA
MODEL_VERSION=vicuna-v1-5-7b

# # pre-train the linear projection
# deepspeed llava/train/train_mem.py \
#     --deepspeed ./scripts/zero2.json \
#     --model_name_or_path lmsys/vicuna-7b-v1.5 \
#     --version mlp2x_gelu \
#     --data_path /home/gs3260/gysun/datasets/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
#     --image_folder /home/gs3260/gysun/datasets/LLaVA-Pretrain/images \
#     --vision_tower openai/clip-vit-large-patch14 \
#     --mm_projector_type linear \
#     --tune_mm_mlp_adapter True \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --bf16 True \
#     --output_dir ./checkpoints/llava-$MODEL_VERSION-linear-pretrain \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 32 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 24000 \
#     --max_steps 16 \
#     --save_total_limit 1 \
#     --learning_rate 2e-3 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True \
#     --report_to wandb \
#     --run_name 'swi-7b-pretrain-linear'

# pre-train the re-sampler

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version plain \
    --data_path /home/gs3260/gysun/datasets/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder /home/gs3260/gysun/datasets/LLaVA-Pretrain/images \
    --vision_tower openai/clip-vit-large-patch14 \
    --mm_projector_type perceiver \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/llava-$MODEL_VERSION-resampler-pretrain \
    --num_train_epochs 5 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --max_steps 16 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name 'swi-7b-pretrain-resampler'