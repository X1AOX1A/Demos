import argparse
from PIL import Image
from minigpt4.common.config import Config
from minigpt4.common.registry import registry


def parse_args():
    parser = argparse.ArgumentParser(description="Image Caption Inference")
    parser.add_argument(
        "--cfg-path", 
        default="/root/Documents/DEMOS/MiniGPT-4/eval_configs/minigpt4_eval.yaml", 
        help="path to configuration file.")
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
    # Input
    IMAGE_PATH = "/root/Documents/DEMOS/OFA/X1A/inference/cat.jpg"
    PROMPT = "Write a detailed description."
    GPU_ID = 0

    print('Initializing Model...')

    # prepare the config
    args = parse_args()
    cfg = Config(args)
    model_config = cfg.model_cfg
    model_config.device_8bit = GPU_ID
    model_cls = registry.get_model_class(model_config.arch)

    # prepare the model
    model = model_cls.from_config(model_config).to('cuda:{}'.format(GPU_ID))
    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    # prepare the image
    raw_image = Image.open(IMAGE_PATH).convert("RGB")
    image = vis_processor["eval"](raw_image).unsqueeze(0).to('cuda:{}'.format(GPU_ID))
    print(model.generate({"image": image, "prompt": PROMPT}))