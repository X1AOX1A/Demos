import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from fairseq import utils, tasks
from fairseq import checkpoint_utils
import sacremoses

import sys
sys.path.append('/root/Documents/DEMOS/OFA')
from tasks.mm_tasks.caption import CaptionTask


# Construct input for caption task
def construct_sample(image: Image, task):
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
    
    # Image transform
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
    
    patch_image = patch_resize_transform(image).unsqueeze(0)
    patch_mask = torch.tensor([True])
    src_text = encode_text(" what does the image describe?", append_bos=True, append_eos=True).unsqueeze(0)
    src_length = torch.LongTensor([s.ne(pad_idx).long().sum() for s in src_text])
    sample = {
        "id":np.array(['42']),
        "net_input": {
            "src_tokens": src_text,
            "src_lengths": src_length,
            "patch_images": patch_image,
            "patch_masks": patch_mask
        }
    }
    return sample
  
# Function to turn FP32 to FP16
def apply_half(t):
    if t.dtype is torch.float32:
        return t.to(dtype=torch.half)
    return t


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
        "scst_args": '{"constraints": "unordered"}',
        "generation": {"constraints": "unordered"}
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

    sample = construct_sample(image, task)
    sample = utils.move_to_cuda(sample) if use_cuda else sample
    sample = utils.apply_to_sample(apply_half, sample) if use_fp16 else sample

    tokenizer = task.build_tokenizer(cfg.tokenizer)
    bpe = task.build_bpe(cfg.bpe)

    def encode_fn(x):
        if tokenizer is not None:
            x = tokenizer.encode(x)
        if bpe is not None:
            x = bpe.encode(x)
        return x

    # Construct constraints
    constraints_list = [["bed", "quilt"]]
    constraints_list = [[task.bpe.encode(c) for c in constraint] for constraint in constraints_list]
    constraints_list = [[task.tgt_dict.encode_line(
        constraint, 
        append_eos=False,
        add_if_not_exist=False
        )
        for constraint in constraints]
        for constraints in constraints_list
    ]
    from fairseq.token_generation_constraints import pack_constraints, unpack_constraints
    constraints_tensor = pack_constraints(constraints_list)

    # Run eval step for caption
    from utils.eval_utils import eval_caption
    with torch.no_grad():
        from torch import tensor
        result, scores = eval_caption(task, generator, models, sample, constraints=constraints_tensor)

    print('Caption: {}'.format(result[0]['caption']))
    # Caption: a kitten laying on top of a wooden tablebedquilt