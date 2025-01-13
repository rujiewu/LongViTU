export PYTHONPATH=$(pwd):$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:256

VERSION="qwen"
PREV_STAGE_CHECKPOINT="checkpoints/Cambrian_Qwen2_7B"
PATH_TO_FOLDER="dataset/longvu"
PATH_TO_JSON="dataset/longvu/misc/longvitu_train_101k.json"

CUDA_LAUNCH_BLOCKING=1 TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --standalone --nproc_per_node=8 --master_port=12345 longvu/train.py \
--output_dir "output" \
--input_model_filename $PREV_STAGE_CHECKPOINT \
--output_model_filename "output/cambrian_qwen2_7b_longvitu_train_101k" \
--data_path $PATH_TO_JSON \
--image_folder $PATH_TO_FOLDER \
--model_max_length 8192 \
--fp16 False \
--bf16 True \
--log_on_each_node False \
--logging_dir "logs" \
--num_train_epochs 1 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 1 \
--save_steps 2000 \
--eval_steps 2000 \
--logging_steps 10 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--report_to "tensorboard" \
--save_total_limit 10 \
--learning_rate 5e-7 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--tf32 False \
--version $VERSION \
--mm_vision_select_layer "-2" \
--mm_use_im_start_end False \
--mm_use_im_patch_token False \
--image_aspect_ratio pad \
--group_by_modality_length True \
--dataloader_num_workers 0 \
--lazy_preprocess True \
--tune_mm_mlp_adapter False \
--freeze_mm_mlp_adapter False \
--freeze_backbone False \
--fsdp "full_shard auto_wrap" \
--fsdp_transformer_layer_cls_to_wrap 'Qwen2DecoderLayer' \
--gradient_checkpointing True \
--mm_projector_type sva \
--image_token_len 144 \
--query_num_list "[144]" \
--resume True \
--lowres_token 8 \
--video_fps 1 \
--highres True \
--drop_threshold 0.8 \