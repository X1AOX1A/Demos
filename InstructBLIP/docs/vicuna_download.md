
# Prepare Vicuna Weights

## 0. Create a temporary environment

```shell
# Optional, Create a new conda environment
conda create -n Vicuna python=3.8
conda activate Vicuna

# Install fastchat to get Vicuna weights
pip3 install fschat
```

## 1. Obtain LLaMA weights and convert to HuggingFace format

- Weights for the LLaMA models can be obtained from by filling out [this form](https://docs.google.com/forms/d/e/1FAIpQLSfqNECQnMkycAp2jP4Z9TFX0cGR4uf7b_fBxjY_OjhJILlKGA/viewform?usp=send_form)

- or, use [pyllama](https://github.com/juncongmoo/pyllama)

```shell
pip install pyllama -U
python -m llama.download \
    # --model_size 7B,13B,30B,65B \
    --model_size 7B,13B \
    # --folder /path/to/download/llama
    --folder /root/Documents/MODELS/LLaMA
```

- After downloading the weights, they will need to be converted to the Hugging Face Transformers format using the [conversion script](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py). The script can be called with the following (example) command:

- [Reference](https://github.com/lm-sys/FastChat)

```shell
# Convert LLaMA weights to HuggingFace format
# For LLaMA-7B
python src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --model_size 7B \
    # --input_dir /path/to/downloaded/llama-7B \
    --input_dir /root/Documents/MODELS/LLaMA/7B \
    # --output_dir /path/to/hf/llama-7b
    --output_dir /root/Documents/MODELS/LLaMA-HF/7B

# For LLaMA-13B
python src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --model_size 13B \
    # --input_dir /path/to/downloaded/llama-13B \
    --input_dir /root/Documents/MODELS/LLaMA/13B \
    # --output_dir /path/to/hf/llama-13b
    --output_dir /root/Documents/MODELS/LLaMA-HF/13B
```

## 2. Get Vicuna weights by applying delta to LLaMA weights

- We release Vicuna weights as delta weights to comply with the LLaMA model license. You can add our delta to the original LLaMA weights to obtain the Vicuna weights. 

- [Reference](https://github.com/lm-sys/FastChat#vicuna-weights)

```shell
# Get Vicuna weights by applying delta to LLaMA weights
# For Vicuna-7B
python3 -m fastchat.model.apply_delta \
    # --base-model-path /path/to/hf/llama-7b \
    --base-model-path /root/Documents/MODELS/LLaMA-HF/7B \
    # --target-model-path /path/to/output/vicuna-7b \
    --target-model-path /root/Documents/MODELS/Vicuna/7B \
    --delta-path lmsys/vicuna-7b-delta-v1.1

# For Vicuna-13B
python3 -m fastchat.model.apply_delta \
    # --base-model-path /path/to/hf/llama-13b \
    --base-model-path /root/Documents/MODELS/LLaMA-HF/13B \
    # --target-model-path /path/to/output/vicuna-13b \
    --target-model-path /root/Documents/MODELS/Vicuna/13B \
    --delta-path lmsys/vicuna-13b-delta-v1.1
```