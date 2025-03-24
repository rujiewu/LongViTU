#!/bin/bash
export OMP_NUM_THREADS=8
export PYTHONWARNINGS="ignore"
export HF_HUB_ENABLE_HF_TRANSFER="1"

export WANDB_ENTITY="<WANDB_ENTITY>"
export WANDB_PROJECT="llava-ov"
export WANDB_MODE="online"

wandb login ${WANDB_API_KEY}
wandb online

LLM_VERSION="Qwen/Qwen2-7B-Instruct"
LLM_VERSION_CLEAN=$(echo "${LLM_VERSION#*/}" | tr '[:upper:]' '[:lower:]')
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"

PROMPT_VERSION="qwen_1_5"
DATASET="longvitu_train_101k"
RUN_NAME="${LLM_VERSION_CLEAN}-${DATASET}-lr1e8-vtlr2e9"
PREV_STAGE_CHECKPOINT="checkpoints/LLaVA-Video-7B-Qwen2"

echo "Previous Checkpoint: ${PREV_STAGE_CHECKPOINT}"
echo "Run Name: ${RUN_NAME}"

NODE_RANK=${NODE_RANK:-1}
MASTER_ADDR=${MASTER_ADDR:-"<MASTER_ADDR>"}
MASTER_PORT=${MASTER_PORT:-12345}

torchrun --nproc_per_node=8 --nnodes=1 --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path $PREV_STAGE_CHECKPOINT \
    --version $PROMPT_VERSION \
    --data_path exps/${DATASET}.yaml \
    --video_folder longvu \
    --mm_tunable_parts "mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr 2e-9 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $RUN_NAME \
    --output_dir output/llava-ov/$RUN_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 10 \
    --learning_rate 1e-8 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --frames_upbound 64 \
    --mm_newline_position grid \
    --add_time_instruction True \
    --force_sample True \
    --mm_spatial_pool_stride 2