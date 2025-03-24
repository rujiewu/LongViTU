from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
import requests
import copy
import warnings
from decord import VideoReader, cpu
import numpy as np
warnings.filterwarnings("ignore")

import datetime
import json
import os
import re
import uuid
from itertools import chain
import argparse
import random

from transformers import logging
logging.set_verbosity_error()

import sys
# sys.path.append('./')
import pandas as pd

import torch
from torch import distributed as dist
from tqdm import tqdm
from decord import cpu, VideoReader  # @manual=fbsource//third-party/pypi/decord:decord

from transformers.trainer_pt_utils import IterableDatasetShard

class EvalDataset(torch.utils.data.IterableDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str,
    ) -> None:
        super(EvalDataset, self).__init__()

        self.data = json.load(open(os.path.join(data_path, "test.json"), "r"))
    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, i):
        return self.data[i]

def load_video(video_path, max_frames_num, fps=1, force_sample=False):
    # if max_frames_num == 0:
    #     return np.zeros((1, 336, 336, 3))
    if not os.path.exists(video_path):
        print(f"!!! video not exist !!!   >>>{video_path}<<<")
        return np.zeros((1, 336, 336, 3)).astype(np.uint8), None, None

    vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)

    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps()/fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i/fps for i in frame_idx]

    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]

    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames, frame_time, video_time

def train(args) -> None:
    local_rank = os.environ['LOCAL_RANK']
    device = torch.device(f"cuda:{local_rank}")
    print(f"device: {device}")
    torch.cuda.set_device(device)

    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=8))

    model_name = "llava_qwen"
    model_path = args.model_path

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        None,
        model_name,
        torch_dtype="bfloat16",
        device_map={"": f"cuda:{local_rank}"},
    )
    
    model.config.use_cache = True
    model.cuda()

    dataset = EvalDataset(data_path=args.data_path)

    world_size = dist.get_world_size()
    world_rank = dist.get_rank()
    shard_dataset = IterableDatasetShard(
        dataset,
        batch_size=1,
        num_processes=world_size,
        process_index=world_rank,
    )

    dist.barrier()
    output = []
    final_output = [None] * world_size

    for line in tqdm(shard_dataset):
        key = line["key"]
        video_info = line["video_info"]
        question_data = line["qa"]
        question = question_data["question"]
        answer = question_data["answer"]
        task_type = question_data["question_type"][0]

        qs = f"Question: {question}\nRespond with only the letter (A, B, C or D) of the correct option."
        
        video_path = os.path.join(args.data_path, "videos", f"{key}.mp4")
        
        max_frames_num = 64
        video, frame_time, video_time = load_video(video_path, max_frames_num, 1, force_sample=True)
        video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().bfloat16()
        video = [video]

        if getattr(model.config, "mm_use_im_start_end", False):
            qs = (
                DEFAULT_IM_START_TOKEN
                + DEFAULT_IMAGE_TOKEN
                + DEFAULT_IM_END_TOKEN
                + "\n"
                + qs
            )
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv_template = "qwen_1_5"
        conv = conv_templates[conv_template].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
        
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=video,
                modalities= ["video"],
                do_sample=False,
                max_new_tokens=5,
                use_cache=True,
            )
        if isinstance(output_ids, tuple):
            output_ids = output_ids[0]
        pred = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        ans = str(pred)

        pred = pred.replace("Answer", "")

        letters = ["A", "B", "C", "D"]
        pred_answer = re.findall("[\(\ ]*[A-D][\)\ ]*", pred)
        if not pred_answer:
            pred_idx = random.choice([0, 1, 2, 3])
            pred = letters[pred_idx]
        else:
            pred_answer = pred_answer[0].strip()
            pred_answer = pred_answer.strip("()")
            if pred_answer in letters:
                pred_idx = letters.index(pred_answer)
                pred = letters[pred_idx]
            else:
                print("pred_answer: ", pred_answer, " pred: ", pred, flush=True)
                pred_idx = random.choice([0, 1, 2, 3])
                pred = letters[pred_idx]

        ans_id = uuid.uuid4()
        output.append({
            "key": key,
            "question": question,
            "prompt": qs,
            "answer": answer,
            "ans": ans,
            "pred": pred,
            "task_type": task_type,
            "answer_id": str(ans_id),
            "model_name": model_name,
        })

    dist.barrier()
    dist.all_gather_object(
        final_output,
        output,
    )
    all_output = list(chain(*final_output))

    global_rank = dist.get_rank()
    if global_rank == 0:
        json.dump(all_output, open(os.path.join(args.data_path, "results", f"{args.model_path.replace('output/', '').replace('/', '_').replace('_consolidated', '')}_outputs.json"), "w"), indent=4)

        correct = 0
        total = 0
        acc_dict = {}
        for output in all_output:
            pred = output["pred"]
            gt = output["answer"]
            task_type = output["task_type"]
            if task_type not in acc_dict:
                acc_dict[task_type] = [0, 0]
            acc_dict[task_type][1] += 1
            total += 1

            if pred == gt:
                acc_dict[task_type][0] += 1
                correct += 1

        final_res = dict()
        total = 0
        idx = 0
        for k, v in acc_dict.items():
            idx += 1
            final_res[k] = v[0] / v[1] * 100
            total += final_res[k]
        final_res["Acc"] = total / idx
        print(final_res, flush=True)

        json.dump(final_res, open(os.path.join(args.data_path, "results", f"{args.model_path.replace('output/', '').replace('/', '_').replace('_consolidated', '')}_result.json"), "w"), indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', default="checkpoints/LLaVA-Video-7B-Qwen2")
    parser.add_argument('--data_path', default="ood/lvbench")
    args = parser.parse_args()

    train(args)