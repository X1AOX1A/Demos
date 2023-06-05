# Download MiniGPT-4

## 1. Prepare the pretrained Vicuna-V0 weights

- The current version of MiniGPT-4 is built on the v0 versoin of Vicuna-13B. Please refer to [our instruction](https://github.com/X1AOX1A/Demos/blob/main/Vicuna/download_vicuna.md) here to prepare the Vicuna weights. 

- The final weights would be in a single folder (e.g., `PATH_TO_VICUNA_V0/7B` or `PATH_TO_VICUNA_V0/13B`) in a structure similar to the following:

    ```shell
    # export PATH_TO_VICUNA_V0="/root/Documents/MODELS/Vicuna-V0"
    PATH_TO_VICUNA_V0
    |-- [ 334]  13B
    |   |-- [ 565]  config.json
    |   |-- [ 137]  generation_config.json
    |   |-- [9.3G]  pytorch_model-00001-of-00003.bin
    |   |-- [9.2G]  pytorch_model-00002-of-00003.bin
    |   |-- [6.1G]  pytorch_model-00003-of-00003.bin
    |   |-- [ 33K]  pytorch_model.bin.index.json
    |   |-- [ 411]  special_tokens_map.json
    |   |-- [488K]  tokenizer.model
    |   `-- [ 727]  tokenizer_config.json
    `-- [ 290]  7B
        |-- [ 564]  config.json
        |-- [ 137]  generation_config.json
        |-- [9.3G]  pytorch_model-00001-of-00002.bin
        |-- [3.3G]  pytorch_model-00002-of-00002.bin
        |-- [ 26K]  pytorch_model.bin.index.json
        |-- [ 411]  special_tokens_map.json
        |-- [488K]  tokenizer.model
        `-- [ 727]  tokenizer_config.json
    ```

- Then, set the path to the vicuna weight in the model config file [here](https://github.com/X1AOX1A/Demos/blob/main/MiniGPT-4/minigpt4/configs/models/minigpt4.yaml#L16) at Line 16.

## 2. Prepare the pretrained MiniGPT-4 checkpoint

- Download the pretrained MiniGPT-4 checkpoint from Google Drive:

```
export PATH_TO_MiniGPT4="/root/Documents/MODELS/MiniGPT-4"

## Download MiniGPT-4 checkpoint aligned with Vicuna 7B
# Download mannually and upload to $PATH_TO_MiniGPT4/7B
# https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R

## Download MiniGPT-4 checkpoint aligned with Vicuna 13B
# Download mannually and upload to $PATH_TO_MiniGPT4/13B
# https://drive.google.com/file/d/1a4zLvaiDBr-36pasffmgpvH5P7CKmpze
```

- The Vicuna weights are organized as follows:

    ```shell
    PATH_TO_MiniGPT4
    |-- [  45]  13B
    |   `-- [ 45M]  pretrained_minigpt4.pth
    `-- [  47]  7B
        `-- [ 36M]  pretrained_minigpt4_7b.pth
    ```

- Then, set the path to the pretrained checkpoint in the evaluation config file in [eval_configs/minigpt4_eval.yaml](https://github.com/X1AOX1A/Demos/blob/main/MiniGPT-4/eval_configs/minigpt4_eval.yaml#L10) at Line 11.