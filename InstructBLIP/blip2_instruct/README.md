# InstructBLIP

## Prepare COCO Caption

- Ref to [Datasets-MS_COCO](https://github.com/X1AOX1A/Datasets/tree/main/MS_COCO)

## Prepare Vicuna Weights

- Ref to [Demos-Vicuna](https://github.com/X1AOX1A/Demos/tree/main/Vicuna)

## Image Caption Inference

```shell
cd infer
python image_caption.py
```

- output:

    ```shell
    ["The image depicts a small cat sitting underneath a wooden bed frame, peeking out from behind it. The cat appears to be curious and interested in what's happening around it. There are two other cats in the scene, one on the left side of the image and the other on the right side. Both cats seem to be relaxing and enjoying their surroundings. In addition to the cats, there is a person visible in the background, possibly taking care of them or interacting with them."]
    ```

## Finetune InstructBLIP on COCO Caption

```shell
bash train/train_caption_coco.sh
```

- It may take 5 hours on 4 40GB A100 GPUs.

- output: (with bug)

    ```shell
    {"test": 
        {
            "Bleu_1": 0.6456479384971744, 
            "Bleu_2": 0.5052420130194599, 
            "Bleu_3": 0.3843327099490886, 
            "Bleu_4": 0.2888257389065465, 
            "METEOR": 0.31081583936668, 
            "ROUGE_L": 0.5325931803724581, 
            "CIDEr": 0.9149957264609108, 
            "SPICE": 0.2477929364222591
        }
    }
    ```