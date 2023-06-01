# BLIP2

## Prepare COCO Caption

- Ref to [Datasets-MS_COCO](https://github.com/X1AOX1A/Datasets/tree/main/MS_COCO)

## Image Caption Inference

## COCO Test Samples

<details open></summary>COCO Test Samples</summary>

```shell
{"image": "val2014/COCO_val2014_000000391895.jpg", 
 "caption": ["A man with a red helmet on a small moped on a dirt road. ", 
            "Man riding a motor bike on a dirt road on the countryside.", 
            "A man riding on the back of a motorcycle.", 
            "A dirt path with a young person on a motor bike rests to the foreground of a verdant area with a bridge and a background of cloud-wreathed mountains. ", 
            "A man in a red shirt and a red hat is on a motorcycle on a hill side."]}, 
{"image": "val2014/COCO_val2014_000000060623.jpg", 
 "caption": ["A young girl inhales with the intent of blowing out a candle. ", 
            "A young girl is preparing to blow out her candle.", 
            "A kid is to blow out the single candle in a bowl of birthday goodness. ", 
            "Girl blowing out the candle on an ice-cream ", 
            "A little girl is getting ready to blow out a candle on a small dessert."]}, 
{"image": "val2014/COCO_val2014_000000483108.jpg", 
 "caption": ["A man on a bicycle riding next to a train", 
            "A person is riding a bicycle but there is a train in the background.", 
            "a red and white train and a man riding a bicycle", 
            "a guy that is riding his bike next to a train", 
            "A man riding a bike past a train traveling along tracks."]}, 
{"image": "val2014/COCO_val2014_000000384213.jpg",
 "caption": ["A kitchen is shown with a variety of items on the counters.", 
            "A kitchen has the windows open and plaid curtains.", 
            "A kitchen with two windows and two metal sinks.", 
            "An older kitchen with cluttered counter tops but empty sink.", 
            "Glasses and bottles are placed near a kitchen sink."]}, 
{"image": "val2014/COCO_val2014_000000386164.jpg",
 "caption": ["A wooden ball on top of a wooden stick.", 
            "The table is full of wooden spoons and utensils.", 
            "A wood table holding an assortment of wood cooking utensils.", 
            "A selection of wooden kitchen tools on a counter.", 
            "Wooden spoons are lined up on a table"]}, 
{"image": "val2014/COCO_val2014_000000223648.jpg",
 "caption": ["Multiple wooden spoons are shown on a table top.", 
            "A table surrounded by chairs and filled with cooking utensils.", 
            "Wooden spoons laid out across a kitchen table.", 
            "Wooden spoons and forks are all over a table.", 
            "A table and chairs with wooden kitchen tools on top."]}
```
</details>

## Zero-shot on COCO Caption

```shell
bash eval/eval_caption_coco_zeroshot.sh
```

- results (id: 20230530192):

    ```shell
    {"test": {"Bleu_1": 0.8087646987513475, "Bleu_2": 0.6635921266407677, "Bleu_3": 0.5253239038842143, "Bleu_4": 0.4083353178203791, "METEOR": 0.30675232411996856, "ROUGE_L": 0.5995837387260223, "CIDEr": 1.364507624426746, "SPICE": 0.2463983715446396}}
    ```

- output examples (test_epochbest.json):

    ```shell
    {"caption": "a man riding a motorcycle on a dirt road", "image_id": 391895}, 
    {"caption": "wooden utensils on a table", "image_id": 386164}, 
    {"caption": "a group of people riding bicycles on a city street", "image_id": 462565}, 
    {"caption": "black and white photo of a man on a motorcycle", "image_id": 559665}, 
    {"caption": "bunches of bananas in a wooden box", "image_id": 579664}, 
    {"caption": "a small blue and white airplane flying in the grass", "image_id": 561100}, 
    {"caption": "a table with chairs and a vase of sunflowers in front of a window", "image_id": 165547}, 
    ```

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

- or directly evaluate the CoCo fine-tuned checkpoint:

```
bash eval/eval_caption_coco.sh
```

- results (id: 20230530191):

    ```shell
    {"test": {"Bleu_1": 0.8201297151042333, "Bleu_2": 0.6798741833125047, "Bleu_3": 0.5447682489351027, "Bleu_4": 0.4282143073884025, "METEOR": 0.3206602458928521, "ROUGE_L": 0.6190080234427047, "CIDEr": 1.4555016501050175, "SPICE": 0.2553750024016312}}
    ```

- output examples (test_epochbest.json):

    ```shell
    {"caption": "a person riding a motorcycle on a dirt road", "image_id": 391895}, 
    {"caption": "wooden spoons and forks are arranged on a table", "image_id": 386164}, 
    {"caption": "a group of people riding bikes down a city street", "image_id": 462565}, 
    {"caption": "a black and white photo of two men on a motorcycle", "image_id": 559665}, 
    {"caption": "a bunch of bananas sitting on top of each other", "image_id": 579664}, 
    {"caption": "a small blue and white plane sitting on top of a grass field", "image_id": 561100}, 
    ```