import os

from numpy import true_divide
import gradio as gr
import warnings

warnings.filterwarnings("ignore")

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
import vqa
import cv2
from PIL import Image
import numpy as np

# Use this command for evaluate the GLIP-T model
config_file = "configs/glip_Swin_T_O365_GoldG.yaml"
weight_file = "checkpoints/glip_tiny_model_o365_goldg_cc_sbu.pth"

# manual override some options
cfg.local_rank = 0
cfg.num_gpus = 1
cfg.merge_from_file(config_file)
cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

glip_demo = GLIPDemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
    show_mask_heatmaps=False
)
blip_demo = vqa.VQA(
    model_path = 'checkpoints/model_base_vqa_capfilt_large.pth')

def predict_image(image, object, question):
    result, _ = glip_demo.run_on_web_image(image[:, :, [2, 1, 0]], object, 0.5)
    result = result[:, :, [2, 1, 0]]
    answer = blip_demo.vqa_demo(image, question)
    return result, answer

def predict_video(video, object, question, frame_drop_value):
    vid = cv2.VideoCapture(video)
    count = 0
    while True:
        ret, frame = vid.read()
        if ret:
            count+=1
            if count % frame_drop_value == 0:
                # image = Image.fromarray(frame)
                image = frame
                cv2.putText(
                img = image,
                text = str(count),
                org = (20, 20),
                fontFace = cv2.FONT_HERSHEY_DUPLEX,
                fontScale = 0.5,
                color = (125, 246, 55),
                thickness = 1)
                result, _ = glip_demo.run_on_web_image(image[:, :, [2, 1, 0]], object, 0.5)
                answer = blip_demo.vqa_demo(image, question)
                yield result, answer
        else:
            break

    yield result, answer

with gr.Blocks() as demo:
    gr.Markdown("Text-Based Object Detection and Visual Question Answering")
    with gr.Tab("Image"):
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(label='input image')
                obj_input = gr.Textbox(label='Objects', lines=1, placeholder="Objects here..")
                vqa_input = gr.Textbox(label='Question', lines=1, placeholder="Question here..")
                image_button = gr.Button("Submit")

            with gr.Column():
                image_output = gr.outputs.Image(type="pil", label="grounding results")
                vqa_output = gr.Textbox(label="Answer")
        
    with gr.Tab("Video"):
        with gr.Row():
            with gr.Column():
                video_input = gr.PlayableVideo(label='input video', mirror_webcam=False)
                obj_input_video = gr.Textbox(label='Objects', lines=1, placeholder="Objects here..")
                vqa_input_video = gr.Textbox(label='Question', lines=1, placeholder="Question here..")
                frame_drop_input = gr.Slider(label='Frames drop value', minimum=0, maximum=30, step=1, value=5)
                video_button = gr.Button("Submit")

            with gr.Column():
                video_output = gr.outputs.Image(type="pil", label="grounding results")
                vqa_output_video = gr.Textbox(label="Answer")
        
    image_button.click(predict_image, inputs=[image_input, obj_input, vqa_input], outputs=[image_output, vqa_output])
    video_button.click(predict_video, inputs=[video_input, obj_input_video, vqa_input_video, frame_drop_input], outputs=[video_output, vqa_output_video])

demo.queue()
demo.launch()