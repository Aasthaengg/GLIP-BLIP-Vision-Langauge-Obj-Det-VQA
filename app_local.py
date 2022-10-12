import os
import gradio as gr
import warnings

warnings.filterwarnings("ignore")

os.system("python setup.py build develop --user")

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
import vqa

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
    model_path = 'checkpoints/model_base_vqa_capfilt_large.pth'
)

def predict(image, object, question):
    result, _ = glip_demo.run_on_web_image(image[:, :, [2, 1, 0]], object, 0.5)
    answer = blip_demo.vqa_demo(image, question)
    return result[:, :, [2, 1, 0]], answer

image = gr.inputs.Image()

gr.Interface(
    description="GLIP + BLIP VQA Demo.",
    fn=predict,
    inputs=[
        "image", 
        gr.Textbox(label='Objects', lines=1, placeholder="Objects here.."), 
        gr.Textbox(label='Question', lines=1, placeholder="Question here..")],

    outputs=[
        gr.outputs.Image(
            type="pil",
            label="grounding results"
        ),
        gr.Textbox(label="Answer")
    ],
).launch()