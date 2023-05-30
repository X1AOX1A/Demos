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
<p align="center">
<img src="infer/cat.jpg" alt="cat" style="width:50%;"> 
</p>

- output:

    `["The image depicts a small cat sitting underneath a wooden bed frame, peeking out from behind it. The cat appears to be curious and interested in what's happening around it. There are two other cats in the scene, one on the left side of the image and the other on the right side. Both cats seem to be relaxing and enjoying their surroundings. In addition to the cats, there is a person visible in the background, possibly taking care of them or interacting with them."]`
    
    

## Finetune InstructBLIP on COCO Caption

```shell
bash train/train_caption_coco.sh
```

- It may take 5 hours on 4 40GB A100 GPUs (with frozen VIT).

### 1. Config1

```shell
model:
  arch: blip2_vicuna_instruct
  model_type: vicuna7b        
  load_finetuned: False

  # vit encoder
  # InstructBLIP maintains the same image resolution (224×224) during instruction
  # tuning and keeps the visual encoder frozen during ﬁnetuning.
  image_size: 224               # follow blip2_instruct_vicuna7b.yaml (pre-train)
  use_grad_checkpoint: True
  vit_precision: "fp16"
  freeze_vit: True

datasets:
  coco_caption_extend: # name of the dataset builder
    vis_processor:
        train:
          name: "blip2_image_train"
          image_size: 224       # follow blip2_instruct_vicuna7b.yaml (pre-train)
        eval:
          name: "blip_image_eval"
          image_size: 224       # follow blip2_instruct_vicuna7b.yaml (pre-train)
    text_processor:
        train:
          name: "blip_caption"
          prompt: "a photo of " # follow blip2_caption_coco_ft.yaml (fine-tune)
        eval:
          name: "blip_caption"
```

- results (id: 20230530073):

    ```shell
    {"val": {"Bleu_1": 0.647760605064115, "Bleu_2": 0.5081890057061402, "Bleu_3": 0.3870170587289628, "Bleu_4": 0.2912790733097921, "METEOR": 0.3105995652934542, "ROUGE_L": 0.5343182267923948, "CIDEr": 0.9134107450064268, "SPICE": 0.24478299750963978}}
    {"val": {"Bleu_1": 0.5973055018941694, "Bleu_2": 0.46719563440119, "Bleu_3": 0.35508273509355626, "Bleu_4": 0.26709520067779924, "METEOR": 0.3025066755240536, "ROUGE_L": 0.5138951656222592, "CIDEr": 0.7688301034126004, "SPICE": 0.23116755154918991}}
    {"val": {"Bleu_1": 0.5770818611224684, "Bleu_2": 0.4514493446288603, "Bleu_3": 0.3441983234504983, "Bleu_4": 0.2605700851723652, "METEOR": 0.3009256915964947, "ROUGE_L": 0.5065751216446777, "CIDEr": 0.7280729940998132, "SPICE": 0.22763485904351846}}
    {"val": {"Bleu_1": 0.5771319500030715, "Bleu_2": 0.45014913368789566, "Bleu_3": 0.34284573914070754, "Bleu_4": 0.25943214491020883, "METEOR": 0.30091932290952034, "ROUGE_L": 0.5060713644392506, "CIDEr": 0.715442978792575, "SPICE": 0.22842278087565324}}
    {"val": {"Bleu_1": 0.5768921741925045, "Bleu_2": 0.44874855467959657, "Bleu_3": 0.3416551869039863, "Bleu_4": 0.258942834298103, "METEOR": 0.30060672479425654, "ROUGE_L": 0.5055461980659608, "CIDEr": 0.7206900767766667, "SPICE": 0.22744325720067235}}
    {"test": {"Bleu_1": 0.6456479384971744, "Bleu_2": 0.5052420130194599, "Bleu_3": 0.3843327099490886, "Bleu_4": 0.2888257389065465, "METEOR": 0.31081583936668, "ROUGE_L": 0.5325931803724581, "CIDEr": 0.9149957264609108, "SPICE": 0.2477929364222591}}
    ```
- output examples (test_epochbest.json):

    ```shell
    {"caption": "a photo of a man riding a motorcycle down a dirt road with a mountain in the background", "image_id": 391895}, 
    {"caption": "a collection of wooden spoons, forks, and other kitchen utensils are lined up on a table", "image_id": 386164}, 
    {"caption": "a group of people riding bicycles down a bike lane on a city street", "image_id": 462565}, 
    {"caption": "a black and white photograph of a man riding a motorcycle on the road", "image_id": 559665}, 
    {"caption": "a photo of bananas are piled high in a wooden crate at an outdoor market", "image_id": 579664}, 
    {"caption": "a close up of a small blue and white airplane sitting on top of a grassy field", "image_id": 561100},
    ```

### 2. Config2

```shell
model:
  arch: blip2_vicuna_instruct
  model_type: vicuna7b        
  load_finetuned: False

  # vit encoder
  image_size: 224
  use_grad_checkpoint: True
  vit_precision: "fp16"
  freeze_vit: True

datasets:
  coco_caption_extend: # name of the dataset builder
    vis_processor:
        train:
          name: "blip2_image_train"
          image_size: 224
        eval:
          name: "blip_image_eval"
          image_size: 224
    text_processor:
        train:
          name: "blip_caption"
          # prompt: "a photo of " # remove prompt
        eval:
          name: "blip_caption"
```
- results (id: 20230530140):

    ```shell
    {"val": {"Bleu_1": 0.7220016618963974, "Bleu_2": 0.5704259859815827, "Bleu_3": 0.438630491578156, "Bleu_4": 0.3326873871582765, "METEOR": 0.31416461167510695, "ROUGE_L": 0.5639709605355904, "CIDEr": 1.1862728592930292, "SPICE": 0.25501800646269157}}
    {"val": {"Bleu_1": 0.7270454898307603, "Bleu_2": 0.5730622702537845, "Bleu_3": 0.44045212811716666, "Bleu_4": 0.33428713049895437, "METEOR": 0.3150557488214792, "ROUGE_L": 0.5637670404562377, "CIDEr": 1.201592559395042, "SPICE": 0.2533526675385253}}
    {"val": {"Bleu_1": 0.726243748354819, "Bleu_2": 0.5731382656596921, "Bleu_3": 0.4418103304017568, "Bleu_4": 0.3358400135261264, "METEOR": 0.31564573729292555, "ROUGE_L": 0.5656436362650309, "CIDEr": 1.1995867636091258, "SPICE": 0.2538992603065989}}

    ```

- output examples (test_epochbest.json):

    ```shell
    ```

### Config3

```shell
model:
  arch: blip2_vicuna_instruct
  model_type: vicuna7b        
  load_finetuned: False

  # vit encoder
  image_size: 364           # 224 -> 364, same as BLIP2-OPT
  use_grad_checkpoint: True
  vit_precision: "fp32"     # fp16 -> fp32, same as BLIP2-OPT
  freeze_vit: True

datasets:
  coco_caption_extend: # name of the dataset builder
    vis_processor:
        train:
          name: "blip2_image_train"
          image_size: 364   # 224 -> 364, same as BLIP2-OPT
        eval:
          name: "blip_image_eval"
          image_size: 364   # 224 -> 364, same as BLIP2-OPT
    text_processor:
        train:
          name: "blip_caption"
        eval:
          name: "blip_caption"
```

- results (id: ):

    ```shell
    ```

- output examples (test_epochbest.json):

    ```shell
    ```