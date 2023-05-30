# BLIP2

## Prepare COCO Caption

- Ref to [Datasets-MS_COCO](https://github.com/X1AOX1A/Datasets/tree/main/MS_COCO)

## Image Caption Inference

## Zero-shot on COCO Caption

## Fine-tune on COCO Caption

```shell
bash train/train_caption_coco.sh
```

- It may take 15 hours on 4 40GB A100 GPUs.

```shell
model:
  arch: blip2_opt
  model_type: caption_coco_opt2.7b
  load_finetuned: False
  use_grad_checkpoint: True
  freeze_vit: False

datasets:
  coco_caption: # name of the dataset builder
    vis_processor:
        train:
          name: "blip2_image_train"
          image_size: 364
        eval:
          name: "blip_image_eval"
          image_size: 364
    text_processor:
        train:
          name: "blip_caption"
          prompt: "a photo of "
        eval:
          name: "blip_caption"
```

- results:

    ```shell
    {"val": {"Bleu_1": 0.7847680890537888, "Bleu_2": 0.6372136659364951, "Bleu_3": 0.49869384838487024, "Bleu_4": 0.38295671768133194, "METEOR": 0.3091472757948791, "ROUGE_L": 0.5959486403703204, "CIDEr": 1.3192180976353052, "SPICE": 0.24957230296952515}}
    {"val": {"Bleu_1": 0.810981062337285, "Bleu_2": 0.6677594627032021, "Bleu_3": 0.5309568205164307, "Bleu_4": 0.4156272491769148, "METEOR": 0.3120647849185179, "ROUGE_L": 0.6070380466362483, "CIDEr": 1.3875641592069825, "SPICE": 0.24706001122123616}}
    ```

- output examples (val_epoch1.json):

    ```shell
    {"caption": "a boy with an umbrella petting a cow", "image_id": 184613}, 
    {"caption": "a bathroom with two sinks and a fire hydrant", "image_id": 340559}, 
    {"caption": "a bathroom with a bath tub and a sink", "image_id": 472621}, 
    {"caption": "a tower with a clock on top of it", "image_id": 462341}, 
    {"caption": "a blue and yellow airplane with a red propeller", "image_id": 78371}, 
    {"caption": "a small red and white plane flying in the sky", "image_id": 9426}, 
    ```