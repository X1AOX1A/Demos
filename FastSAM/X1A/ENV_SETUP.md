# Enviornment Setup

```shell
conda create -n FastSAM python=3.9
conda activate FastSAM
cd FastSAM

# Install the packages
pip install -r requirements.txt

# optional, since mirrors.sustech does not have latest packages
pip install gradio==3.35.2 --index-url=https://pypi.org/simple/
pip install ultralytics==8.0.120 --index-url=https://pypi.org/simple/

# Install CLIP
pip install git+https://github.com/openai/CLIP.git
```

- [Reference](https://github.com/CASIA-IVA-Lab/FastSAM#installation)