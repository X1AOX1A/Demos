# MiniGPT-4

## 0. Preparation

### Prepare COCO Caption

- Ref to [Datasets-MS_COCO](https://github.com/X1AOX1A/Datasets/tree/main/MS_COCO)

<details></summary>COCO Test Samples</summary>

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

### Prepare Vicuna Weights

- Ref to [Demos-Vicuna](https://github.com/X1AOX1A/Demos/tree/main/Vicuna)

### Prepare MiniGPT-4 Weights

- Ref to [Demos-MiniGPT-4](https://github.com/X1AOX1A/Demos/blob/main/MiniGPT-4/X1A/download_minigpt4.md)

## 1. Image Caption Inference

```shell
cd infer
python image_caption.py # prompt: "Write a detailed description."
```
<p align="center">
<img src="infer/cat.jpg" alt="cat" style="width:50%;"> 
</p>

- output:

    `This is an image of a small cat peeking out from under a bed. It appears to be a domestic shorthair cat with grey and white fur, standing on its hind legs and looking up at the camera. It has bright blue eyes and a small, pointed face. The cat is standing on a wooden bed frame, with a blanket draped over it. The background appears to be a wall with a white sheet hanging on it. The lighting in the room is soft and diffused, casting a warm glow on the cat's fur.`

## 2. Zero-shot on COCO Caption

```shell
bash X1A/eval/caption_coco_zeroshot_vicuna13b.sh
```

- results (id: ):

    ```shell
    ```

- output examples (test_epochbest.json):

    ```shell
    ```

## 3. Fine-tune on COCO Caption


```shell
bash X1A/train/caption_coco_ft_vicuna13b.sh
```

- results (id: ):

    ```shell
    ```

- output examples (test_epochbest.json):

    ```shell
    ```