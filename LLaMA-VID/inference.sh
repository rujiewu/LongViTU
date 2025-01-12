CUDA_VISIBLE_DEVICES=0 python llamavid/serve/run_llamavid_movie.py --model-path work_dirs/llama-vid-7b-finetune-224-short-video-stage2base --eval finetune --load-4bit --video-file data/ood/egoschema

CUDA_VISIBLE_DEVICES=1 python llamavid/serve/run_llamavid_movie.py --model-path work_dirs/llama-vid-7b-finetune-224-short-video-stage2base --eval finetune --load-4bit --video-file data/ood/videomme

CUDA_VISIBLE_DEVICES=2 python llamavid/serve/run_llamavid_movie.py --model-path work_dirs/llama-vid-7b-finetune-224-short-video-stage2base --eval finetune --load-4bit --video-file data/ood/worldqa

CUDA_VISIBLE_DEVICES=3 python llamavid/serve/run_llamavid_movie.py --model-path work_dirs/llama-vid-7b-finetune-224-short-video-stage2base --eval finetune --load-4bit --video-file data/ood/openeqa