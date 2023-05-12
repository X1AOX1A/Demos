#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
caption raw text with a trained model. Batches data on-the-fly.
"""

import ast
import fileinput
import logging
import math
import os
import sys
import time
from argparse import Namespace
from collections import namedtuple

import numpy as np
import torch

import torch
import string
import numpy as np
from PIL import Image
from typing import List
from torchvision import transforms
from fairseq import utils, tasks
from fairseq import checkpoint_utils
import sacremoses

import sys
sys.path.append('/root/Documents/DEMOS/OFA')
from tasks.mm_tasks.caption import CaptionTask

from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils
from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.token_generation_constraints import pack_constraints, unpack_constraints
from fairseq_cli.generate import get_symbols_to_strip_from_output

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.interactive")


def buffered_read(input, buffer_size):
    buffer = []
    with fileinput.input(files=[input], openhook=fileinput.hook_encoded("utf-8")) as h:
        for src_str in h:
            buffer.append(src_str.strip())
            if len(buffer) >= buffer_size:
                yield buffer
                buffer = []

    if len(buffer) > 0:
        yield buffer


def construct_sample(images, batch_constraints, cfg, task, max_positions, encode_fn):
    def encode_fn_target(x):
        return encode_fn(x)
    
    # Constraints preprocess
    def encode_constraints(batch_constraints):
        # Convert each List[str] to List[Tensor]
        for i, constraint_list in enumerate(batch_constraints):
            batch_constraints[i] = [
                task.target_dictionary.encode_line(
                    encode_fn_target(constraint),
                    append_eos=False,
                    add_if_not_exist=False,
                )
                for constraint in constraint_list
            ]
            return batch_constraints
        
    # Image transform
    def encode_images(images):
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

        patch_images = []
        for image in images:
            patch_image = patch_resize_transform(image)
            patch_images.append(patch_image)

        patch_masks = torch.tensor([True for _ in range(len(images))])
        return patch_images, patch_masks

    batch_size = len(images)
    ids = list(range(batch_size))
    # Encode constraints
    if cfg.generation.constraints:
        batch_constraints = encode_constraints(batch_constraints)
        constraints_tensor = pack_constraints(batch_constraints)
    else:
        constraints_tensor = None
    # Encode prompts
    lines = [" what does the image describe?" for _ in range(batch_size)]
    # TODO: Check whether lines(prompts) add bos and eos
    src_tokens, src_lengths = task.get_interactive_tokens_and_lengths(lines, encode_fn)
    # Encode images
    # TODO: Check images size [1, 3, 480, 480]
    patch_images, patch_masks = encode_images(images)

    return {
        "ids": ids,
        "src_tokens": torch.stack(src_tokens),
        "src_lengths": torch.tensor(src_lengths),
        "patch_images": torch.stack(patch_images),
        "patch_masks": patch_masks,
        "constraints": constraints_tensor,
    }

# Function to turn FP32 to FP16
def apply_half(t):
    if t.dtype is torch.float32:
        return t.to(dtype=torch.half)
    return t


def main():

    MODEL_PATH = "/root/Documents/MODELS/OFA/Finetuned_OFA_Base_COCO/caption_base_best.pt"
    IMAGE_PATH = "/root/Documents/DEMOS/OFA/X1A/inference/cat.jpg"

    images = [Image.open(IMAGE_PATH)]
    constraints = [["bed", "quilt"]]
    

    # Register refcoco task
    tasks.register_task('caption', CaptionTask)

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

    start_time = time.time()
    total_caption_time = 0

    utils.import_user_module(cfg.common)

    if cfg.interactive.buffer_size < 1:
        cfg.interactive.buffer_size = 1
    if cfg.dataset.max_tokens is None and cfg.dataset.batch_size is None:
        cfg.dataset.batch_size = 1

    assert (
        not cfg.generation.sampling or cfg.generation.nbest == cfg.generation.beam
    ), "--sampling requires --nbest to be equal to --beam"
    logger.info(cfg)

    # Fix seed for stochastic decoding
    if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    # turn on cuda if GPU is available
    use_cuda = torch.cuda.is_available()
    # use fp16 only when GPU is available
    use_fp16 = True if use_cuda else False

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    # Optimize ensemble for generation
    for model in models:
        if model is None:
            continue
        if cfg.common.fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    # Initialize generator
    generator = task.build_generator(models, cfg.generation)

    # Handle tokenization and BPE
    tokenizer = task.build_tokenizer(cfg.tokenizer)
    bpe = task.build_bpe(cfg.bpe)

    def encode_fn(x):
        if tokenizer is not None:
            x = tokenizer.encode(x)
        if bpe is not None:
            x = bpe.encode(x)
        return x

    def decode_fn(x):
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(cfg.generation.replace_unk)

    max_positions = utils.resolve_max_positions(
        task.max_positions(), *[model.max_positions() for model in models]
    )

    if cfg.generation.constraints:
        logger.warning(
            "NOTE: Constrained decoding currently assumes a shared subword vocabulary."
        )

    if cfg.interactive.buffer_size > 1:
        logger.info("Sentence buffer size: %s", cfg.interactive.buffer_size)
    logger.info("NOTE: hypothesis and token scores are output in base 2")

    results = []
    sample =  construct_sample(images, constraints, cfg, task, max_positions, encode_fn)
    ids = sample["ids"]
    batch_size = len(ids)
    src_tokens = sample["src_tokens"]
    src_lengths = sample["src_lengths"]
    patch_images = sample["patch_images"]
    patch_masks = sample["patch_masks"]
    constraints = sample["constraints"]

    sample = {
        "id": ids,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
            "patch_images": patch_images,
            "patch_masks": patch_masks,
        },
    }

    sample = utils.move_to_cuda(sample) if use_cuda else sample
    sample = utils.apply_to_sample(apply_half, sample) if use_fp16 else sample
    constraints = constraints.cuda() if use_cuda else constraints

    caption_start_time = time.time()
    captions = task.inference_step(
        generator, models, sample, constraints=constraints
    )
    caption_time = time.time() - caption_start_time
    total_caption_time += caption_time
    list_constraints = [[] for _ in range(batch_size)]
    if cfg.generation.constraints:
        list_constraints = [unpack_constraints(c) for c in constraints]
    for i, (id, hypos) in enumerate(zip(sample["id"], captions)):
        src_tokens_i = utils.strip_pad(src_tokens[i], tgt_dict.pad())
        constraints = list_constraints[i]
        results.append(
            (
                id,
                src_tokens_i,
                hypos,
                {
                    "constraints": constraints,
                    "time": caption_time / len(captions),
                },
            )
        )

    # sort output to match input order
    for id_, src_tokens, hypos, info in sorted(results, key=lambda x: x[0]):
        src_str = ""
        if src_dict is not None:
            src_str = src_dict.string(src_tokens, cfg.common_eval.post_process)
            print("S-{}\t{}".format(id_, src_str))
            print("W-{}\t{:.3f}\tseconds".format(id_, info["time"]))
            for constraint in info["constraints"]:
                print(
                    "C-{}\t{}".format(
                        id_,
                        tgt_dict.string(constraint, cfg.common_eval.post_process),
                    )
                )

        # Process top predictions
        for hypo in hypos[: min(len(hypos), cfg.generation.nbest)]:
            # hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
            #     hypo_tokens=hypo["tokens"].int().cpu(),
            #     src_str=src_str,
            #     alignment=hypo["alignment"],
            #     align_dict=align_dict,
            #     tgt_dict=tgt_dict,
            #     remove_bpe=cfg.common_eval.post_process,
            #     extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator),
            # )
            hypo_str = hypo["tokens"]
            detok_hypo_str = decode_fn(hypo_str)
            score = hypo["score"] / math.log(2)  # convert to base 2
            # original hypothesis (after tokenization and BPE)
            print("H-{}\t{}\t{}".format(id_, score, hypo_str))
            # detokenized hypothesis
            print("D-{}\t{}\t{}".format(id_, score, detok_hypo_str))
            print(
                "P-{}\t{}".format(
                    id_,
                    " ".join(
                        map(
                            lambda x: "{:.4f}".format(x),
                            # convert from base e to base 2
                            hypo["positional_scores"].div_(math.log(2)).tolist(),
                        )
                    ),
                )
            )
            if cfg.generation.print_alignment:
                alignment_str = " ".join(
                    ["{}-{}".format(src, tgt) for src, tgt in alignment]
                )
                print("A-{}\t{}".format(id_, alignment_str))


    logger.info(
        "Total time: {:.3f} seconds; caption time: {:.3f}".format(
            time.time() - start_time, total_caption_time
        )
    )


if __name__ == "__main__":
    main()
