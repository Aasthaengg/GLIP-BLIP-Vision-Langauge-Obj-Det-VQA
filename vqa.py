import sys
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from models.blip_vqa import blip_vqa
import cv2
import numpy as np
import matplotlib.image as mpimg

from skimage import transform as skimage_transform
from scipy.ndimage import filters
from matplotlib import pyplot as plt


import torch
from torch import nn
from torchvision import transforms

import json
import traceback

class VQA:
    def __init__(self, model_path, image_size=480):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = blip_vqa(pretrained=model_path, image_size=image_size, vit='base')
        self.block_num = 9
        self.model.eval()
        self.model.text_encoder.base_model.base_model.encoder.layer[self.block_num].crossattention.self.save_attention = True

        self.model = self.model.to(self.device)
    def getAttMap(self, img, attMap, blur = True, overlap = True):
        attMap -= attMap.min()
        if attMap.max() > 0:
            attMap /= attMap.max()
        attMap = skimage_transform.resize(attMap, (img.shape[:2]), order = 3, mode = 'constant')
        if blur:
            attMap = filters.gaussian_filter(attMap, 0.02*max(img.shape[:2]))
            attMap -= attMap.min()
            attMap /= attMap.max()
        cmap = plt.get_cmap('jet')
        attMapV = cmap(attMap)
        attMapV = np.delete(attMapV, 3, 2)
        if overlap:
            attMap = 1*(1-attMap**0.7).reshape(attMap.shape + (1,))*img + (attMap**0.7).reshape(attMap.shape+(1,)) * attMapV
        return attMap

    def gradcam(self, text_input, image_path, image):
        mask = text_input.attention_mask.view(text_input.attention_mask.size(0),1,-1,1,1)
        grads = self.model.text_encoder.base_model.base_model.encoder.layer[self.block_num].crossattention.self.get_attn_gradients()
        cams = self.model.text_encoder.base_model.base_model.encoder.layer[self.block_num].crossattention.self.get_attention_map()
        cams = cams[:, :, :, 1:].reshape(image.size(0), 12, -1, 30, 30) * mask
        grads = grads[:, :, :, 1:].clamp(0).reshape(image.size(0), 12, -1, 30, 30) * mask
        gradcam = cams * grads
        gradcam = gradcam[0].mean(0).cpu().detach()

        num_image = len(text_input.input_ids[0])
        num_image -= 1
        fig, ax = plt.subplots(num_image, 1, figsize=(15,15*num_image))

        rgb_image = cv2.imread(image_path)[:, :, ::-1]
        rgb_image = np.float32(rgb_image) / 255
        ax[0].imshow(rgb_image)
        ax[0].set_yticks([])
        ax[0].set_xticks([])
        ax[0].set_xlabel("Image")

        for i,token_id in enumerate(text_input.input_ids[0][1:-1]):
            word = self.model.tokenizer.decode([token_id])
            gradcam_image = self.getAttMap(rgb_image, gradcam[i+1])
            ax[i+1].imshow(gradcam_image)
            ax[i+1].set_yticks([])
            ax[i+1].set_xticks([])
            ax[i+1].set_xlabel(word)
        
        plt.show()


    def load_demo_image(self, image_size, img_path, device):
        raw_image = Image.open(img_path).convert('RGB')   
        w,h = raw_image.size
        transform = transforms.Compose([
            transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ]) 
        image = transform(raw_image).unsqueeze(0).to(device)   
        return raw_image, image

    def vqa(self, img_path, question):
        raw_image, image = self.load_demo_image(image_size=480, img_path=img_path, device=self.device)        
        answer, vl_output, que = self.model(image, question, mode='gradcam', inference='generate')
        loss = vl_output[:,1].sum()
        self.model.zero_grad()
        loss.backward()

        with torch.no_grad():
            self.gradcam(que, img_path, image)
        
        return answer[0]

    def vqa_demo(self, image, question):
        image_size = 480
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ]) 
        image = transform(image).unsqueeze(0).to(self.device)
        answer = self.model(image, question, mode='inference', inference='generate')
        
        return answer[0]


if __name__=="__main__":
    if not len(sys.argv) == 3:
        print('Format: python3 vqa.py <path_to_img> <question>')
        print('Sample: python3 vqa.py sample.jpg "What is the color of the horse?"')
        
    else:
        model_path = 'checkpoints/model_base_vqa_capfilt_large.pth'
        vqa_object = VQA(model_path=model_path)
        img_path = sys.argv[1]
        question = sys.argv[2]
        answer = vqa_object.vqa(img_path, question)
        print('Question: {} | Answer: {}'.format(question, answer))