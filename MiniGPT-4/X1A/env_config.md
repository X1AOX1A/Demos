# Enviornment Setup

```shell
cd ~/Documents/DEMOS
git clone https://github.com/Vision-CAIR/MiniGPT-4.git
cd MiniGPT-4
conda env create -f environment.yml
conda activate minigpt4

# Update torch
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

- [Ref](https://github.com/Vision-CAIR/MiniGPT-4#installation)

Add `setup.py` to MiniGPT-4:

```python
from setuptools import setup, find_namespace_packages
setup(
    name='minigpt4',
    version='1.0',
    author='Your Name',
    author_email='your_email@example.com',
    description='A brief description of your project',
    install_requires=[],
    packages=find_namespace_packages(include="minigpt4.*"),
)

```

Then install the minigpt4 package:

```shell
pip install -e .
```