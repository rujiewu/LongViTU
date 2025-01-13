# import spaces

import os
import re
import traceback

import torch
import gradio as gr

import sys

import numpy as np

from longvu.builder import load_pretrained_model
from longvu.constants import (
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from longvu.conversation import conv_templates, SeparatorStyle
from longvu.mm_datautils import (
    KeywordsStoppingCriteria,
    process_images,
    tokenizer_image_token,
)
from decord import cpu, VideoReader


title_markdown = """
<div style="display: flex; justify-content: center; align-items: center; text-align: center;">
  <div>
    <h1 >LongVU: Spatiotemporal Adaptive Compression for Long Video-Language Understanding</h1>
  </div>
</div>
<div align="center">
    <div style="display:flex; gap: 0.25rem; margin-top: 10px;" align="center">
        <a href='https://vision-cair.github.io/LongVU/'><img src='https://img.shields.io/badge/Project-LongVU-blue'></a>
        <a href='https://huggingface.co/Vision-CAIR/LongVU_Qwen2_7B'><img src='https://img.shields.io/badge/model-checkpoints-yellow'></a>
    </div>
</div>
"""

block_css = """
#buttons button {
    min-width: min(120px,100%);
    color: #9C276A
}
"""

plum_color = gr.themes.colors.Color(
    name='plum',
    c50='#F8E4EF',
    c100='#E9D0DE',
    c200='#DABCCD',
    c300='#CBA8BC',
    c400='#BC94AB',
    c500='#AD809A',
    c600='#9E6C89',
    c700='#8F5878',
    c800='#804467',
    c900='#713056',
    c950='#662647',
)


class Chat:

    def __init__(self):
        self.version = "qwen"
        model_name = "cambrian_qwen"
        model_path = "./checkpoints/longvu_qwen"
        device = "cuda:7"

        self.tokenizer, self.model, self.processor, _ = load_pretrained_model(model_path,  None, model_name, device=device)
        self.model.eval()

    def remove_after_last_dot(self, s):
        last_dot_index = s.rfind('.')
        if last_dot_index == -1:
            return s
        return s[:last_dot_index + 1]
    
    # @spaces.GPU(duration=120)
    @torch.inference_mode()
    def generate(self, data: list, message, temperature, top_p, max_output_tokens):
        # TODO: support multiple turns of conversation.
        assert len(data) == 1

        tensor, image_sizes, modal = data[0]

        conv = conv_templates[self.version].copy()

        if isinstance(message, str):
            conv.append_message("user", DEFAULT_IMAGE_TOKEN + '\n' + message)
        elif isinstance(message, list):
            if DEFAULT_IMAGE_TOKEN not in message[0]['content']:
                message[0]['content'] = DEFAULT_IMAGE_TOKEN + '\n' + message[0]['content']
            for mes in message:
                conv.append_message(mes["role"], mes["content"])

        conv.append_message("assistant", None)
        
        prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(self.model.device)
        )

        if "llama3" in self.version:
            input_ids = input_ids[0][1:].unsqueeze(0)  # remove bos
        
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=tensor,
                image_sizes=image_sizes,
                do_sample=True,
                temperature=temperature,
                max_new_tokens=max_output_tokens,
                use_cache=True,
                top_p=top_p,
                stopping_criteria=[stopping_criteria],
            )
        
        pred = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
        return self.remove_after_last_dot(pred)


# @spaces.GPU(duration=120)
def generate(image, video, message, chatbot, textbox_in, temperature, top_p, max_output_tokens, dtype=torch.float16):
    if textbox_in is None:
        raise gr.Error("Chat messages cannot be empty")
        return (
            gr.update(value=image, interactive=True),
            gr.update(value=video, interactive=True),
            message,
            chatbot,
            None,
        )
    
    data = []

    processor = handler.processor
    try:
        if image is not None:
            data.append((processor['image'](image).to(handler.model.device, dtype=dtype), None, '<image>'))
        elif video is not None:
            vr = VideoReader(video, ctx=cpu(0), num_threads=1)
            fps = float(vr.get_avg_fps())
            frame_indices = np.array(
                [
                    i
                    for i in range(
                        0,
                        len(vr),
                        round(fps),
                    )
                ]
            )
            video_tensor = []
            for frame_index in frame_indices:
                img = vr[frame_index].asnumpy()
                video_tensor.append(img)
            video_tensor = np.stack(video_tensor)
            image_sizes = [video_tensor[0].shape[:2]]
            video_tensor = process_images(video_tensor, processor, handler.model.config)
            video_tensor = [item.unsqueeze(0).to(handler.model.device, dtype=dtype) for item in video_tensor]
            data.append((video_tensor, image_sizes, '<video>'))
        elif image is None and video is None:
            data.append((None, None, '<text>'))
        else:
            raise NotImplementedError("Not support image and video at the same time")
    except Exception as e:
        traceback.print_exc()
        return gr.update(value=None, interactive=True), gr.update(value=None, interactive=True), message, chatbot, None

    assert len(message) % 2 == 0, "The message should be a pair of user and system message."

    show_images = ""
    if image is not None:
        show_images += f'<img src="./file={image}" style="display: inline-block;width: 250px;max-height: 400px;">'
    if video is not None:
        show_images += f'<video controls playsinline width="300" style="display: inline-block;"  src="./file={video}"></video>'

    one_turn_chat = [textbox_in, None]

    # 1. first run case
    if len(chatbot) == 0:
        one_turn_chat[0] += "\n" + show_images
    # 2. not first run case
    else:
        # scanning the last image or video
        length = len(chatbot)
        for i in range(length - 1, -1, -1):
            previous_image = re.findall(r'<img src="./file=(.+?)"', chatbot[i][0])
            previous_video = re.findall(r'<video controls playsinline width="500" style="display: inline-block;"  src="./file=(.+?)"', chatbot[i][0])

            if len(previous_image) > 0:
                previous_image = previous_image[-1]
                # 2.1 new image append or pure text input will start a new conversation
                if (video is not None) or (image is not None and os.path.basename(previous_image) != os.path.basename(image)):
                    message.clear()
                    one_turn_chat[0] += "\n" + show_images
                break
            elif len(previous_video) > 0:
                previous_video = previous_video[-1]
                # 2.2 new video append or pure text input will start a new conversation
                if image is not None or (video is not None and os.path.basename(previous_video) != os.path.basename(video)):
                    message.clear()
                    one_turn_chat[0] += "\n" + show_images
                break

    message.append({'role': 'user', 'content': textbox_in})
    text_en_out = handler.generate(data, message, temperature=temperature, top_p=top_p, max_output_tokens=max_output_tokens)
    message.append({'role': 'assistant', 'content': text_en_out})

    one_turn_chat[1] = text_en_out
    chatbot.append(one_turn_chat)

    return gr.update(value=image, interactive=True), gr.update(value=video, interactive=True), message, chatbot, None


def regenerate(message, chatbot):
    message.pop(-1), message.pop(-1)
    chatbot.pop(-1)
    return message, chatbot


def clear_history(message, chatbot):
    message.clear(), chatbot.clear()
    return (gr.update(value=None, interactive=True),
            gr.update(value=None, interactive=True),
            message, chatbot,
            gr.update(value=None, interactive=True))

handler = Chat()

textbox = gr.Textbox(show_label=False, placeholder="Enter text and press ENTER", container=False)

theme = gr.themes.Default(primary_hue=plum_color)
# theme.update_color("primary", plum_color.c500)
theme.set(slider_color="#9C276A")
theme.set(block_title_text_color="#9C276A")
theme.set(block_label_text_color="#9C276A")
theme.set(button_primary_text_color="#9C276A")

with gr.Blocks(title='LongVU', theme=theme, css=block_css) as demo:
    gr.Markdown(title_markdown)
    message = gr.State([])

    with gr.Row():
        with gr.Column(scale=3):
            image = gr.State(None)
            video = gr.Video(label="Input Video")

            with gr.Accordion("Parameters", open=True) as parameter_row:

                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.2,
                    step=0.1,
                    interactive=True,
                    label="Temperature",
                )

                top_p = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.7,
                        step=0.1,
                        interactive=True,
                        label="Top P",
                )

                max_output_tokens = gr.Slider(
                    minimum=64,
                    maximum=512,
                    value=128,
                    step=64,
                    interactive=True,
                    label="Max output tokens",
                )

        with gr.Column(scale=7):
            chatbot = gr.Chatbot(label="LongVU", bubble_full_width=True, height=420)
            with gr.Row():
                with gr.Column(scale=8):
                    textbox.render()
                with gr.Column(scale=1, min_width=50):
                    submit_btn = gr.Button(value="Send", variant="primary", interactive=True)
            with gr.Row(elem_id="buttons") as button_row:
                upvote_btn     = gr.Button(value="👍  Upvote", interactive=True)
                downvote_btn   = gr.Button(value="👎  Downvote", interactive=True)
                regenerate_btn = gr.Button(value="🔄  Regenerate", interactive=True)
                clear_btn      = gr.Button(value="🗑️  Clear history", interactive=True)

    with gr.Row():
        with gr.Column():
            gr.Examples(
                examples=[
                    [
                        f"./examples/video3.mp4",
                        "What is the moving direction of the yellow ball?",
                    ],
                    [
                        f"./examples/video1.mp4",
                        "Describe this video in detail.",
                    ],
                    [
                        f"./examples/video2.mp4",
                        "What is the name of the store?",
                    ],
                ],
                inputs=[video, textbox],
            )

    submit_btn.click(
        generate, 
        [image, video, message, chatbot, textbox, temperature, top_p, max_output_tokens],
        [image, video, message, chatbot])

    regenerate_btn.click(
        regenerate, 
        [message, chatbot], 
        [message, chatbot]).then(
        generate, 
        [image, video, message, chatbot, textbox, temperature, top_p, max_output_tokens], 
        [image, video, message, chatbot, textbox])

    textbox.submit(
        generate,
        [
            image,
            video,
            message,
            chatbot,
            textbox,
            temperature,
            top_p,
            max_output_tokens,
        ],
        [image, video, message, chatbot, textbox],
    )
    
    clear_btn.click(
        clear_history, 
        [message, chatbot],
        [image, video, message, chatbot, textbox])

demo.launch()
