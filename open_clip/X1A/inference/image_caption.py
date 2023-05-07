import open_clip
import torch
from PIL import Image

MODEL_DIR = "/root/Documents/MODELS/VIT/coca_ViT-L-14/mscoco_finetuned_laion2B-s13B-b90k"
IMAGE = "/root/Documents/DEMOS/open_clip/X1A/cat.jpg"

# Generating text with CoCa
model, _, transform = open_clip.create_model_and_transforms(
  model_name="coca_ViT-L-14",
  pretrained="mscoco_finetuned_laion2B-s13B-b90k",
  cache_dir=MODEL_DIR
)

im = Image.open(IMAGE).convert("RGB")
im = transform(im).unsqueeze(0)

with torch.no_grad(), torch.cuda.amp.autocast():
  generated = model.generate(im)

print(open_clip.decode(generated[0]).split("<end_of_text>")[0].replace("<start_of_text>", ""))