import argparse
from PIL import Image
from minigpt4.common.config import Config
from minigpt4.common.registry import registry


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
    args.cfg_path = "/root/Documents/DEMOS/MiniGPT-4/X1A/infer/image_caption_13b.yaml"
    args.gpu_id = 0
    
    # Inputs
    IMAGE_PATH = "/root/Documents/DEMOS/OFA/X1A/inference/cat.jpg"
    
    print('Initializing Model...')
    # prepare the model
    cfg = Config(args)
    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)

    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    # prepare the image
    raw_image = Image.open(IMAGE_PATH).convert("RGB")
    image = vis_processor(raw_image).unsqueeze(0).to('cuda:{}'.format(args.gpu_id))
    print(model.generate({"image": image}))