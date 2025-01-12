import json
import torch
import pickle
import argparse

from llamavid.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llamavid.conversation import conv_templates, SeparatorStyle
from llamavid.model.builder import load_pretrained_model
from llamavid.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the command-line arguments
    parser.add_argument("--model-path", type=str, default="work_dirs/llama-vid/llama-vid-7b-full-224-long-video")
    parser.add_argument("--eval", type=str, default="zeroshot")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default='vicuna_v1')
    parser.add_argument("--video-file", type=str, required=True)
    parser.add_argument("--video-token", type=int, default=2)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")

    return parser.parse_args()

def run_inference(args):
    """
    Run inference on ActivityNet QA DataSet using the Video-ChatGPT model.

    Args:
        args: Command-line arguments.
    """
    replace_llama_attn_with_flash_attn(inference=True)

    # Initialize the model
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit)
    
    QAs = json.load(open(f"{args.video_file}/test.json", "r"))
    for qa_idx in range(len(QAs)):
        try:
            print(qa_idx)
            video_file = QAs[qa_idx]["video"]
            video_info = pickle.load(open(f"{args.video_file}/pkls/{video_file}", "rb"))
            input_prompt = video_info['inputs']

            ##### modify #####
            duration = int(video_info['feats'].shape[0])
            if duration <= 900:
                step = 1
            elif 900 < duration <= 1800:
                step = 2
            elif 1800 < duration <= 3600:
                step = 4
            elif 3600 < duration <= 7200:
                step = 8
            
            # step = 1
            video = torch.from_numpy(video_info['feats'][:, 1:])
            video = video[0::step, :, :]

            input_prompt = "<image>" * video.shape[0]
            ##### modify #####
            
            # replace the default image token with multiple tokens
            input_prompt = input_prompt.replace(DEFAULT_IMAGE_TOKEN, DEFAULT_IMAGE_TOKEN * args.video_token)

            ##### modify #####
            # video = torch.from_numpy(video_info['feats'][:, 1:]).cuda().half()
            video = video.cuda().half()
            video = [video]
            ##### modify #####
            
            question = QAs[qa_idx]["conversations"][0]["value"].replace("<image>\n", "").split("\n")[0]
            print(question)
            if model.config.mm_use_im_start_end:
                question = DEFAULT_IM_START_TOKEN + input_prompt + DEFAULT_IM_END_TOKEN + '\n' + question
            else:
                question = input_prompt + '\n' + question

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            # print('> Input token num:', len(input_ids[0]))

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            cur_prompt = str(question)
            with torch.inference_mode():
                model.update_prompt([[cur_prompt]])
                output_ids = model.generate(
                    input_ids,
                    images=video,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                    max_new_tokens=1024,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[:-len(stop_str)]
            outputs = outputs.strip()
            QAs[qa_idx][args.eval] = outputs

            with open(f"{args.video_file}/{args.eval}.json", "w") as file:
                json.dump(QAs, file, indent=4)
        except:
            continue

if __name__ == "__main__":
    args = parse_args()
    run_inference(args)