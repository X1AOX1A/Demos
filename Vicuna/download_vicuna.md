
# Prepare Vicuna Weights

Replace the directories in the following commands with the appropriate paths.

```shell
export PATH_TO_LLAMA="/root/Documents/MODELS/LLaMA"
export PATH_TO_LLAMA_HF="/root/Documents/MODELS/LLaMA-HF"
export PATH_TO_VICUNA_V0="/root/Documents/MODELS/Vicuna-V0"
export PATH_TO_VICUNA_V11="/root/Documents/MODELS/Vicuna-V1.1"
```

## 0. Install dependencies

```shell
# Optional, Create a new conda environment
# functools.cache requires python>=3.9
conda create -n Vicuna python=3.9
conda activate Vicuna
```

## 1. Obtain LLaMA weights

- Weights for the LLaMA models can be obtained from by filling out [this form](https://docs.google.com/forms/d/e/1FAIpQLSfqNECQnMkycAp2jP4Z9TFX0cGR4uf7b_fBxjY_OjhJILlKGA/viewform?usp=send_form)

- or, use ths [BitTorrent link](magnet:?xt=urn:btih:ZXXDAUWYLRUXXBHUYEMS6Q5CE5WA3LVA&dn=LLaMA) ([Reference](https://github.com/facebookresearch/llama/pull/73/files))

    ```shell
    sudo apt-get install transmission-cli
    transmission-cli \
    --download-dir $PATH_TO_LLAMA \
    magnet:?xt=urn:btih:ZXXDAUWYLRUXXBHUYEMS6Q5CE5WA3LVA&dn=LLaMA
    ```

- or, use `download_llama_community.sh` from [pyllama](https://github.com/juncongmoo/pyllama) (**Preferred**)

    ```shell
    bash ./download_llama_community.sh "7B,13B" $PATH_TO_LLAMA
    ```

- The downloaded files are organized as follows:

    ```shell
    PATH_TO_LLAMA
    |-- [ 120]  13B
    |   |-- [ 154]  checklist.chk
    |   |-- [ 12G]  consolidated.00.pth
    |   |-- [ 12G]  consolidated.01.pth
    |   `-- [ 101]  params.json
    |-- [  89]  7B
    |   |-- [ 100]  checklist.chk
    |   |-- [ 13G]  consolidated.00.pth
    |   `-- [ 101]  params.json
    |-- [488K]  tokenizer.model
    `-- [  50]  tokenizer_checklist.chk
    ```

## 2. Convert LLaMA weights to HuggingFace format

- After downloading the weights, they will need to be converted to the Hugging Face Transformers format using the [conversion script](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py). The script can be called with the following (example) command.

- [Reference](https://github.com/lm-sys/FastChat)

    ```shell
    # Convert LLaMA weights to HuggingFace format
    # For LLaMA-7B
    python -m transformers.models.llama.convert_llama_weights_to_hf \
        --model_size 7B \
        --input_dir $PATH_TO_LLAMA \
        --output_dir $PATH_TO_LLAMA_HF/7B

    # For LLaMA-13B
    python -m transformers.models.llama.convert_llama_weights_to_hf \
        --model_size 13B \
        --input_dir $PATH_TO_LLAMA \
        --output_dir $PATH_TO_LLAMA_HF/13B
    ```

- The converted files are organized as follows:

    ```shell
    PATH_TO_LLAMA_HF
    |-- [4.0K]  13B
    |   |-- [ 502]  config.json
    |   |-- [ 132]  generation_config.json
    |   |-- [9.3G]  pytorch_model-00001-of-00003.bin
    |   |-- [9.2G]  pytorch_model-00002-of-00003.bin
    |   |-- [6.1G]  pytorch_model-00003-of-00003.bin
    |   |-- [ 33K]  pytorch_model.bin.index.json
    |   |-- [ 411]  special_tokens_map.json
    |   |-- [1.8M]  tokenizer.json
    |   |-- [488K]  tokenizer.model
    |   `-- [ 727]  tokenizer_config.json
    `-- [ 316]  7B
        |-- [ 502]  config.json
        |-- [ 132]  generation_config.json
        |-- [9.3G]  pytorch_model-00001-of-00002.bin
        |-- [3.3G]  pytorch_model-00002-of-00002.bin
        |-- [ 26K]  pytorch_model.bin.index.json
        |-- [ 411]  special_tokens_map.json
        |-- [1.8M]  tokenizer.json
        |-- [488K]  tokenizer.model
        `-- [ 727]  tokenizer_config.json
    ```

## 3. Get Vicuna weights by applying delta to LLaMA weights

- We release Vicuna weights as delta weights to comply with the LLaMA model license. You can add our delta to the original LLaMA weights to obtain the Vicuna weights. 

- [Reference](https://github.com/lm-sys/FastChat#vicuna-weights)


### Vicuna-V0

- Apply the delta weights to the LLaMA-HF weights:

    ```shell
    # Install fastchat to get Vicuna weights
    # Install the FastChat library that is compatible with v0 Vicuna by
    pip install git+https://github.com/lm-sys/FastChat.git@v0.1.10

    # Get Vicuna weights by applying delta to LLaMA weights
    # For Vicuna-7B-V0
    python -m fastchat.model.apply_delta \
        --base-model-path $PATH_TO_LLAMA_HF/7B \
        --target-model-path $PATH_TO_VICUNA_V0/7B \
        --delta-path lmsys/vicuna-7b-delta-v0

    # For Vicuna-13B-V0
    python -m fastchat.model.apply_delta \
        --base-model-path $PATH_TO_LLAMA_HF/13B \
        --target-model-path $PATH_TO_VICUNA_V0/13B \
        --delta-path lmsys/vicuna-13b-delta-v0
    ```

- The Vicuna weights are organized as follows:

    ```shell
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


### Vicuna-V1.1

- Apply the delta weights to the LLaMA-HF weights:

    ```shell
    # Install fastchat to get Vicuna weights
    # Vicuna weight V1.1 are only compatible with transformers>=4.28.0 and fschat >= 0.2.0
    pip install --upgrade fschat

    # Get Vicuna weights by applying delta to LLaMA weights
    # For Vicuna-7B-V1.1
    python -m fastchat.model.apply_delta \
        --base-model-path $PATH_TO_LLAMA_HF/7B \
        --target-model-path $PATH_TO_VICUNA_V11/7B \
        --delta-path lmsys/vicuna-7b-delta-v1.1

    # For Vicuna-13B-V1.1
    python -m fastchat.model.apply_delta \
        --base-model-path $PATH_TO_LLAMA_HF/13B \
        --target-model-path $PATH_TO_VICUNA_V11/13B \
        --delta-path lmsys/vicuna-13b-delta-v1.1
    ```

- The Vicuna weights are organized as follows:

    ```shell
    PATH_TO_VICUNA_V11
    |-- [ 334]  13B
    |   |-- [ 560]  config.json
    |   |-- [ 132]  generation_config.json
    |   |-- [9.3G]  pytorch_model-00001-of-00003.bin
    |   |-- [9.2G]  pytorch_model-00002-of-00003.bin
    |   |-- [6.1G]  pytorch_model-00003-of-00003.bin
    |   |-- [ 33K]  pytorch_model.bin.index.json
    |   |-- [ 411]  special_tokens_map.json
    |   |-- [488K]  tokenizer.model
    |   `-- [ 727]  tokenizer_config.json
    `-- [ 290]  7B
        |-- [ 559]  config.json
        |-- [ 132]  generation_config.json
        |-- [9.3G]  pytorch_model-00001-of-00002.bin
        |-- [3.3G]  pytorch_model-00002-of-00002.bin
        |-- [ 26K]  pytorch_model.bin.index.json
        |-- [ 411]  special_tokens_map.json
        |-- [488K]  tokenizer.model
        `-- [ 727]  tokenizer_config.json
    ```