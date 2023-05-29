from lavis.models import load_model_and_preprocess
import torch
from PIL import Image

if __name__ == "__main__":
    # add path to "/root/Documents/DEMOS/LAVIS/lavis/configs/models/blip2/blip2_instruct_vicuna7b.yaml"
    #   llm_model: "/root/Documents/MODELS/Vicuna/7B"
    IMAGE_PATH = "/root/Documents/DEMOS/OFA/X1A/inference/cat.jpg"

    # setup device to use
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    # load sample image
    raw_image = Image.open(IMAGE_PATH).convert("RGB")
    # display(raw_image.resize((596, 437)))

    # loads InstructBLIP model
    model, vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna7b", is_eval=True, device=device)
    # prepare the image
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
    print(model.generate({"image": image, "prompt": "Write a detailed description."}))
    # "The image depicts a small cat sitting underneath a wooden bed frame, peeking out from behind it. The cat appears to be curious and interested in what's happening around it. There are two other cats in the scene, one on the left side of the image and the other on the right side. Both cats seem to be relaxing and enjoying their surroundings. In addition to the cats, there is a person visible in the background, possibly taking care of them or interacting with them."