import argparse
import torch.backends.cudnn as cudnn

from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

class Inference:
    def __init__(self, args):
        print('Initializing Model...')
        cfg = Config(args)

        model_config = cfg.model_cfg
        model_config.device_8bit = args.gpu_id
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

        vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
        self.chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))        
        print('Initialization Finished')

    def generate(self, image, text, num_beams=1, temperature=1.0):
        chat_state = CONV_VISION.copy()
        img_list = []
        llm_message = self.chat.upload_img(image, chat_state, img_list)
        self.chat.ask(text, chat_state)
        llm_message = self.chat.answer(conv=chat_state, img_list=img_list, num_beams=num_beams, temperature=temperature, max_new_tokens=300, max_length=2000)[0]
        return llm_message
    

def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.cfg_path = "/root/Documents/DEMOS/MiniGPT-4/X1A/infer/image_caption.yaml"
    args.gpu_id = 0

    # Inputs
    IMAGE_PATH = "/root/Documents/DEMOS/MiniGPT-4/X1A/infer/cat.jpg"
    TEXT = "Describe this image in detail."

    # Run VQA without history information
    vqa = Inference(args)
    output = vqa.generate(IMAGE_PATH, TEXT, num_beams=5)
    print(output)
    # "The image shows a small cat sitting on a bed, looking up at the camera with big, curious eyes. The cat's fur is a mix of light and dark brown, and it has white paws and a white patch on its chest. The cat's eyes are a bright blue color, and its whiskers are long and bushy. The cat appears to be healthy and well-fed, with a shiny coat and bright eyes. The background of the image is a wooden bed frame with a blanket and pillow on it. The wall behind the bed is made of plaster and has a few cracks in it. There is a window behind the cat, but it is not visible in the image. The overall impression of the image is one of warmth and comfort, with the cat looking cozy and content on the bed."