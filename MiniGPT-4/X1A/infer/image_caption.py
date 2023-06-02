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


def init_model(args):
    print('Initializing Model...')
    cfg = Config(args)

    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
    print('Initialization Finished')
    return chat


def run_vqa(image, prompt, chat):
    chat_state = CONV_VISION.copy()
    img_list = []
    llm_message = chat.upload_img(image, chat_state, img_list)
    chat.ask(prompt, chat_state)
    llm_message = chat.answer(conv=chat_state, img_list=img_list, num_beams=1, temperature=1.0, max_new_tokens=300, max_length=2000)[0]
    return llm_message


if __name__ == "__main__":
    args = parse_args()
    args.cfg_path = "/root/Documents/DEMOS/MiniGPT-4/eval_configs/minigpt4_eval.yaml"
    args.gpu_id = 0

    # Inputs
    IMAGE_PATH = "/root/Documents/DEMOS/OFA/X1A/inference/cat.jpg"
    PROMPT = "Describe this image in detail."

    # Run VQA
    chat = init_model(args)
    output = run_vqa(IMAGE_PATH, PROMPT, chat)
    print(output)
    # This is an image of a small cat peeking out from under a bed. It appears to be a domestic shorthair cat with grey and white fur, standing on its hind legs and looking up at the camera. It has bright blue eyes and a small, pointed face. The cat is standing on a wooden bed frame, with a blanket draped over it. The background appears to be a wall with a white sheet hanging on it. The lighting in the room is soft and diffused, casting a warm glow on the cat's fur.