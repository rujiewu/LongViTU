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

tasks = {
    "Action Sequence": (
        "action_sequence.json",
        "star/Charades_v1_480/",
        "video",
        True,
    ),  # has start & end
    "Action Prediction": (
        "action_prediction.json",
        "star/Charades_v1_480/",
        "video",
        True,
    ),  # has start & end
    "Action Antonym": ("action_antonym.json", "ssv2_video/", "video", False),
    "Fine-grained Action": (
        "fine_grained_action.json",
        "Moments_in_Time_Raw/videos/",
        "video",
        False,
    ),
    "Unexpected Action": ("unexpected_action.json", "FunQA_test/test/", "video", False),
    "Object Existence": (
        "object_existence.json",
        "clevrer/video_validation/",
        "video",
        False,
    ),
    "Object Interaction": (
        "object_interaction.json",
        "star/Charades_v1_480/",
        "video",
        True,
    ),  # has start & end
    "Object Shuffle": ("object_shuffle.json", "perception/videos/", "video", False),
    "Moving Direction": (
        "moving_direction.json",
        "clevrer/video_validation/",
        "video",
        False,
    ),
    "Action Localization": (
        "action_localization.json",
        "sta/sta_video/",
        "video",
        True,
    ),  # has start & end
    "Scene Transition": ("scene_transition.json", "scene_qa/video/", "video", False),
    "Action Count": ("action_count.json", "perception/videos/", "video", False),
    "Moving Count": ("moving_count.json", "clevrer/video_validation/", "video", False),
    "Moving Attribute": (
        "moving_attribute.json",
        "clevrer/video_validation/",
        "video",
        False,
    ),
    "State Change": ("state_change.json", "perception/videos/", "video", False),
    "Fine-grained Pose": ("fine_grained_pose.json", "nturgbd/", "video", False),
    "Character Order": ("character_order.json", "perception/videos/", "video", False),
    "Egocentric Navigation": ("egocentric_navigation.json", "vlnqa/", "video", False),
    "Episodic Reasoning": (
        "episodic_reasoning.json",
        "tvqa/frames_fps3_hq/",
        "frame",
        True,
    ),  # has start & end, read frame
    "Counterfactual Inference": (
        "counterfactual_inference.json",
        "clevrer/video_validation/",
        "video",
        False,
    ),
}

class EvalDataset(torch.utils.data.IterableDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str,
    ) -> None:
        super(EvalDataset, self).__init__()

        self.data_path = data_path

        list_data_dict = []
        for task_name, task in tasks.items():
            json_file = os.path.join(data_path, "json", task[0])
            vis_folder = os.path.join(data_path, "videos", task[1])
            with open(json_file, "r") as f:
                json_data = json.load(f)
            for data in json_data:
                video_path = os.path.join(vis_folder, data["video"])
                answer = data["answer"]
                question = data["question"]
                answer_idx = -1
                letters = []
                options = data["candidates"]
                options_string = ""
                for option_idx, c in enumerate(options):
                    letters.append(f"{chr(ord('A') + option_idx)}")
                    options_string += f"({chr(ord('A') + option_idx)}) {c}\n"
                    if c == answer:
                        answer_idx = option_idx
                prompt = f"Question: {question}\nOptions:\n{options_string}Answer with the option's letter from the given choices directly and only give the best option."
                list_data_dict.append(
                    {
                        "task_type": task_name,
                        "bound": (data["start"], data["end"]) if task[3] else task[3],
                        "question": question,
                        "prompt": prompt,
                        "answer": answer_idx,
                        "answer_word": data["answer"],
                        "video_name": data["video"].split(".")[0],
                        "video": video_path,
                        "data_type": task[2],
                        "letters": ",".join(letters),
                    }
                )

        self.data = list_data_dict

    def __len__(self) -> int:
        return len(self.data)

    # pyre-fixme[3]: Return type must be annotated.
    def __iter__(self):
        return iter(self.data)

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
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
        video_name = line["video_name"]
        answer = line["answer"]
        qs = line["prompt"]
        task_type = line["task_type"]
        video_path = line["video"]
        bound = line["bound"]
        data_type = line["data_type"]
        letters = line["letters"].split(",")

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
        pred_answer = re.findall(
            f"[\(,\ ]*[{letters[0]}-{letters[-1]}][\),\ ]*", pred
        )
        if not pred_answer:
            pred_idx = random.choice([0, 1, 2])
            try:
                pred = letters[pred_idx]
            except:
                pred = None    
        else:
            pred_answer = pred_answer[0].strip()
            pred_answer = pred_answer.strip("()")
            if pred_answer in letters:
                pred_idx = letters.index(pred_answer)
                pred = letters[pred_idx]
            else:
                print("pred_answer: ", pred_answer, " pred: ", pred, flush=True)
                pred_idx = random.choice([0, 1, 2])
                pred = letters[pred_idx]

        ans_id = uuid.uuid4()
        output.append(
            {
                "question": line["question"],
                "prompt": qs,
                "answer": answer,
                "ans": ans,
                "pred": pred_idx,
                "task_type": task_type,
                "answer_id": str(ans_id),
                "model_id": model_name,
                "video_name": video_name,
            }
        )

    dist.barrier()
    dist.all_gather_object(
        final_output,
        output,
    )
    all_output = list(chain(*final_output))

    global_rank = dist.get_rank()
    if global_rank == 0:
        json.dump(all_output, open(os.path.join(args.data_path, "results", f"{args.model_path.replace('output/', '').replace('/', '_').replace('_consolidated', '')}_outputs.json"), "w"), indent=4)

        task_types = tasks.keys()
        task_acc = {x: [] for x in task_types}
        acc = []

        for i, x in enumerate(all_output):
            value = 1
            if x["pred"] != x["answer"]:
                value = 0
            acc.append(value)
            task_acc[x["task_type"]].append(value)

        acc = sum(acc) * 100 / len(acc)
        task_acc = {x: sum(task_acc[x]) * 100 / len(task_acc[x]) for x in task_acc}
        print(f"Accuracy: ", acc)
        print("Task ccuracy", task_acc)

        task_acc["avg"] = acc

        json.dump(task_acc, open(os.path.join(args.data_path, "results", f"{args.model_path.replace('output/', '').replace('/', '_').replace('_consolidated', '')}_result.json"), "w"), indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', default="checkpoints/LLaVA-Video-7B-Qwen2")
    parser.add_argument('--data_path', default="ood/mvbench")
    args = parser.parse_args()

    train(args)