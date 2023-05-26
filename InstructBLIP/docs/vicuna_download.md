
# Prepare Vicuna Weights

## 1. Obtain LLaMA weights

- Weights for the LLaMA models can be obtained from by filling out [this form](https://docs.google.com/forms/d/e/1FAIpQLSfqNECQnMkycAp2jP4Z9TFX0cGR4uf7b_fBxjY_OjhJILlKGA/viewform?usp=send_form)

- or, use ths [BitTorrent link](magnet:?xt=urn:btih:ZXXDAUWYLRUXXBHUYEMS6Q5CE5WA3LVA&dn=LLaMA) ([Reference](https://github.com/facebookresearch/llama/pull/73/files))

    ```shell
    sudo apt-get install transmission-cli
    transmission-cli \
    --download-dir /root/Documents/MODELS/LLaMA \
    magnet:?xt=urn:btih:ZXXDAUWYLRUXXBHUYEMS6Q5CE5WA3LVA&dn=LLaMA
    ```

- or, use [pyllama](https://github.com/juncongmoo/pyllama)

    ```shell
    pip install pyllama -U
    python -m llama.download \
        --model_size 7B,13B \
        --folder /root/Documents/MODELS/LLaMA
    ```

    - output:

        ```shell
        Downloading tokenizer...
        ✅ /root/Documents/MODELS/LLaMA/tokenizer.model
        ✅ /root/Documents/MODELS/LLaMA/tokenizer_checklist.chk
        tokenizer.model: OK
        Downloading 7B
        downloading file to /root/Documents/MODELS/LLaMA/7B/consolidated.00.pth ...please wait for a few minutes ...
        ✅ /root/Documents/MODELS/LLaMA/7B/consolidated.00.pth
        ✅ /root/Documents/MODELS/LLaMA/7B/params.json
        ✅ /root/Documents/MODELS/LLaMA/7B/checklist.chk
        Checking checksums
        consolidated.00.pth: FAILED
        params.json: OK
        Downloading 13B
        downloading file to /root/Documents/MODELS/LLaMA/13B/consolidated.00.pth ...please wait for a few minutes ...
        ✅ /root/Documents/MODELS/LLaMA/13B/consolidated.00.pth
        downloading file to /root/Documents/MODELS/LLaMA/13B/consolidated.01.pth ...please wait for a few minutes ...
        ✅ /root/Documents/MODELS/LLaMA/13B/consolidated.01.pth
        ✅ /root/Documents/MODELS/LLaMA/13B/params.json
        ✅ /root/Documents/MODELS/LLaMA/13B/checklist.chk
        Checking checksums
        consolidated.00.pth: FAILED
        consolidated.01.pth: OK
        params.json: OK
        ```

- The downloaded files are organized as follows:

    ```shell
    /root/Documents/MODELS/LLaMA
    ├── 7B
    │   ├── checklist.chk
    │   ├── consolidated.00.pth
    │   └── params.json
    ├── 13B
    │   ├── checklist.chk
    │   ├── consolidated.00.pth
    │   ├── consolidated.01.pth
    │   └── params.json
    ├── tokenizer_checklist.chk
    └── tokenizer.model
    ```

## 2. Convert LLaMA weights to HuggingFace format

- After downloading the weights, they will need to be converted to the Hugging Face Transformers format using the [conversion script](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py). The script can be called with the following (example) command.

- [Reference](https://github.com/lm-sys/FastChat)

    ```shell
    # Convert LLaMA weights to HuggingFace format
    # For LLaMA-7B
    python /root/anaconda3/envs/Vicuna/lib/python3.8/site-packages/transformers/models/llama/convert_llama_weights_to_hf.py \
        --model_size 7B \
        --input_dir /root/Documents/MODELS/LLaMA \
        --output_dir /root/Documents/MODELS/LLaMA-HF/7B

    python /root/anaconda3/envs/Vicuna/lib/python3.8/site-packages/transformers/models/llama/convert_llama_weights_to_hf.py \
        --model_size 13B \
        --input_dir /root/Documents/MODELS/LLaMA \
        --output_dir /root/Documents/MODELS/LLaMA-HF/13B
    ```

- The converted files are organized as follows:

    ```shell
    /root/Documents/MODELS/LLaMA-HF
    ├── 7B
    │   ├── config.json
    │   ├── pytorch_model.bin
    │   └── tokenizer.json
    └── 13B
        ├── config.json
        ├── pytorch_model.bin
        └── tokenizer.json
    ```

## 3. Get Vicuna weights by applying delta to LLaMA weights

- We release Vicuna weights as delta weights to comply with the LLaMA model license. You can add our delta to the original LLaMA weights to obtain the Vicuna weights. 

- [Reference](https://github.com/lm-sys/FastChat#vicuna-weights)

    ```shell
    # Optional, Create a new conda environment
    conda create -n Vicuna python=3.8
    conda activate Vicuna

    # Install fastchat to get Vicuna weights
    pip3 install fschat

    # Get Vicuna weights by applying delta to LLaMA weights
    # For Vicuna-7B
    python3 -m fastchat.model.apply_delta \
        --base-model-path /root/Documents/MODELS/LLaMA-HF/7B \
        --target-model-path /root/Documents/MODELS/Vicuna/7B \
        --delta-path lmsys/vicuna-7b-delta-v1.1

    # For Vicuna-13B
    python3 -m fastchat.model.apply_delta \
        --base-model-path /root/Documents/MODELS/LLaMA-HF/13B \
        --target-model-path /root/Documents/MODELS/Vicuna/13B \
        --delta-path lmsys/vicuna-13b-delta-v1.1
    ```

- The Vicuna weights are organized as follows:

    ```shell
    /root/Documents/MODELS/Vicuna
    ├── 7B
    │   ├── config.json
    │   ├── pytorch_model.bin
    │   └── tokenizer.json
    └── 13B
        ├── config.json
        ├── pytorch_model.bin
        └── tokenizer.json
    ```