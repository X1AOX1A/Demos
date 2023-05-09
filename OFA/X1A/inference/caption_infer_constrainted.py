import torch
import string
import numpy as np
from PIL import Image
from typing import List
from torchvision import transforms
from fairseq import utils, tasks
from fairseq import checkpoint_utils

import sys
sys.path.append('/root/Documents/DEMOS/OFA')
from tasks.mm_tasks.caption import CaptionTask


# Construct input for constrainted caption task
def construct_sample(image: Image, constraints_list: List[List[str]], task):
    bos_item = torch.LongTensor([task.src_dict.bos()])
    eos_item = torch.LongTensor([task.src_dict.eos()])
    pad_idx = task.src_dict.pad()

    # Text preprocess
    def encode_text(text, length=None, append_bos=False, append_eos=False):
        s = task.tgt_dict.encode_line(
            line=task.bpe.encode(text),
            add_if_not_exist=False,
            append_eos=False
        ).long()
        if length is not None:
            s = s[:length]
        if append_bos:
            s = torch.cat([bos_item, s])
        if append_eos:
            s = torch.cat([s, eos_item])
        return s
    
    # Constraints preprocess
    def encode_constraints(constraints_list: List[List[str]]):
        constraints_list = [[task.tgt_dict.encode_line(
            task.bpe.encode(constraint), 
            add_if_not_exist=False,
            append_eos=False,
            )
            for constraint in constraints]
            for constraints in constraints_list
        ]
        from fairseq.token_generation_constraints import pack_constraints
        # Add `eos` at begin and `bos` at the end
        constraints_tensor = pack_constraints(constraints_list)
        return constraints_tensor

    # Image transform
    def encode_image(image):
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        patch_resize_transform = transforms.Compose(
            [
                lambda image: image.convert("RGB"),
                transforms.Resize((cfg.task.patch_image_size, cfg.task.patch_image_size), interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        return patch_resize_transform(image)
    
    constraints_tensor = encode_constraints(constraints_list)
    patch_image = encode_image(image).unsqueeze(0)
    patch_mask = torch.tensor([True])
    src_text = encode_text(" what does the image describe?", append_bos=True, append_eos=True).unsqueeze(0)
    src_length = torch.LongTensor([s.ne(pad_idx).long().sum() for s in src_text])    
    sample = {
        "id":np.array(['42']),
        "net_input": {
            "src_tokens": src_text,         # tensor.Size([1, 8])
            "src_lengths": src_length,      # tensor.Size([1])
            "patch_images": patch_image,    # tensor.Size([1, 3, 480, 480])
            "patch_masks": patch_mask       # tensor.Size([1])
        }
    }
    return sample, constraints_tensor
  
# Function to turn FP32 to FP16
def apply_half(t):
    if t.dtype is torch.float32:
        return t.to(dtype=torch.half)
    return t


from utils.eval_utils import decode_fn
def eval_constrainted_caption(task, generator, models, sample, **kwargs):
    transtab = str.maketrans({key: None for key in string.punctuation})
    hypos = task.inference_step(generator, models, sample, **kwargs)
    results = []
    for i, sample_id in enumerate(sample["id"].tolist()):
        detok_hypo_str = decode_fn(hypos[i][0]["tokens"], task.tgt_dict, task.bpe, generator)
        results.append({"image_id": str(sample_id), "caption": detok_hypo_str.translate(transtab).strip()})
    return results, None


if __name__ == "__main__":
    MODEL_PATH = "/root/Documents/MODELS/OFA/Finetuned_OFA_Base_COCO/caption_base_best.pt"
    IMAGE_PATH = "/root/Documents/DEMOS/OFA/X1A/inference/cat.jpg"

    # Register refcoco task
    tasks.register_task('caption', CaptionTask)

    # turn on cuda if GPU is available
    use_cuda = torch.cuda.is_available()
    # use fp16 only when GPU is available
    use_fp16 = True if use_cuda else False

    # Load pretrained ckpt & config
    overrides={
        "eval_cider": False,                    # evaluation with CIDEr scores
        "beam": 5, 
        "max_len_b": 16, 
        "no_repeat_ngram_size": 3, 
        "seed": 7,
        "scst_args": '{"constraints": "ordered"}',
        "generation": {"constraints": "ordered"}
        }
    
    # task -> OFA/tasks/mm_tasks/caption.py/CaptionTask
    #         OFA/tasks/ofa_task.py/OFATask
    #         OFA/fairseq/fairseq/tasks/fairseq_task.py/FairseqTask
    # Initialize task.scst_generator with constraints (cfg.scst_args)
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
            utils.split_paths(MODEL_PATH),
            arg_overrides=overrides
        )

    # Move models to GPU
    for model in models:
        model.eval()
        if use_fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    # Initialize task.scst_generator with constraints (cfg.scst_args)
    generator = task.build_generator(models, cfg.generation)

    image = Image.open(IMAGE_PATH)
    # Note: need to add a blank before the constraint
    constraints_list = [[" cute", " cat"]]

    sample, constraints = construct_sample(image, constraints_list, task)
    sample = utils.move_to_cuda(sample) if use_cuda else sample
    sample = utils.apply_to_sample(apply_half, sample) if use_fp16 else sample
    constraints = constraints.cuda() if use_cuda else constraints
    

    # Run eval step for caption
    with torch.no_grad():
        result, scores = eval_constrainted_caption(
            task, generator, models, sample, constraints=constraints)

    print('Caption: {}'.format(result[0]['caption']))
    # Caption: a cute cat laying on top of a wooden table