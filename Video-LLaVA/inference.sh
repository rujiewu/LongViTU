CUDA_VISIBLE_DEVICES=0 python videollava/eval/video/run_inference_video_qa.py \
    --model_path checkpoints/videollava-7b-finetune-8-memory \
    --cache_dir cache_dir \
    --video_dir ood/egoschema/videos \
    --gt_question_answer ood/egoschema/test.json \
    --output_dir ood/egoschema \
    --output_name finetune-8-memory

CUDA_VISIBLE_DEVICES=1 python videollava/eval/video/run_inference_video_qa.py \
    --model_path checkpoints/videollava-7b-finetune-8-memory \
    --cache_dir cache_dir \
    --video_dir ood/videomme/videos \
    --gt_question_answer ood/videomme/test.json \
    --output_dir ood/videomme \
    --output_name finetune-8-memory

CUDA_VISIBLE_DEVICES=2 python videollava/eval/video/run_inference_video_qa.py \
    --model_path checkpoints/videollava-7b-finetune-8-memory \
    --cache_dir cache_dir \
    --video_dir ood/worldqa/videos \
    --gt_question_answer ood/worldqa/test.json \
    --output_dir ood/worldqa \
    --output_name finetune-8-memory

CUDA_VISIBLE_DEVICES=3 python videollava/eval/video/run_inference_video_qa.py \
    --model_path checkpoints/videollava-7b-finetune-8-memory \
    --cache_dir cache_dir \
    --video_dir ood/openeqa/videos \
    --gt_question_answer ood/openeqa/test.json \
    --output_dir ood/openeqa \
    --output_name finetune-8-memory