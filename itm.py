import sys
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from models.blip_vqa import blip_vqa
from models.blip_itm import blip_itm


class VQA:
    def __init__(self, model_path, image_size=480):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = blip_vqa(pretrained=model_path, image_size=image_size, vit='base')
        self.model.eval()
        self.model = self.model.to(self.device)

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
        with torch.no_grad():
            answer = self.model(image, question, train=False, inference='generate')
            return answer[0]
class ITM:
    def __init__(self, model_path, image_size=384):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = blip_itm(pretrained=model_path, image_size=image_size, vit='base')
        self.model.eval()
        self.model = self.model.to(device='cpu')
    
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

    def itm(self, img_path, caption):
        raw_image, image = self.load_demo_image(image_size=384,img_path=img_path, device=self.device)
        itm_output = self.model(image,caption,match_head='itm')
        itm_score = torch.nn.functional.softmax(itm_output,dim=1)[:,1]
        itc_score = self.model(image,caption,match_head='itc')
        # print('The image and text is matched with a probability of %.4f'%itm_score)
        # print('The image feature and text feature has a cosine similarity of %.4f'%itc_score)
        return itm_score, itc_score

if __name__=="__main__":
    if not len(sys.argv) == 3:
        print('Format: python3 vqa.py <path_to_img> <question>')
        print('Sample: python3 vqa.py sample.jpg "What is the color of the horse?"')
        
    else:
        model_path = 'checkpoints/model_base_vqa_capfilt_large.pth'
        model2_path = 'model_base_retrieval_coco.pth'
        # vqa_object = VQA(model_path=model_path)
        itm_object = ITM(model_path=model2_path)
        img_path = sys.argv[1]
        # question = sys.argv[2]
        caption = sys.argv[2]
        # answer = vqa_object.vqa(img_path, caption)
        itm_score, itc_score = itm_object.itm(img_path, caption)
        # print('Question: {} | Answer: {}'.format(caption, answer))
        print('Caption: {} | The image and text is matched with a probability of %.4f: {} | The image feature and text feature has a cosine similarity of %.4f: {}'.format (caption,itm_score,itc_score))

