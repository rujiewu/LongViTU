#!/bin/bash
export OMP_NUM_THREADS=8

CKPT_DIR="output/llava-ov/qwen2-7b-instruct-longvitu_subset_10k-lr1e8-vtlr2e9"

torchrun --nproc_per_node=4 eval/eval_egoschema.py --data_path ood/egoschema --model_path "$CKPT_DIR"
torchrun --nproc_per_node=4 eval/eval_videomme.py --data_path ood/videomme --model_path "$CKPT_DIR"
torchrun --nproc_per_node=4 eval/eval_mlvu.py --data_path ood/mlvu --model_path "$CKPT_DIR"
torchrun --nproc_per_node=4 eval/eval_lvbench.py --data_path ood/lvbench --model_path "$CKPT_DIR"
torchrun --nproc_per_node=4 eval/eval_mvbench.py --data_path ood/mvbench --model_path "$CKPT_DIR"