# 创建 CLIP 环境

```python
conda create -n CLIP python=3.8
conda activate CLIP

# clip-benchmark 不支持 pytorch2
conda install pytorch==1.13.1 torchvision==0.14.1  pytorch-cuda=11.7 -c pytorch -c nvidia

# 用于下载 MSCOCO 等数据
pip install clip-benchmark

# install jupyter kernel
pip install ipykernel
python -m ipykernel install --user --name CLIP
```

# 从本地代码安装 CLIP

- 下载 [open_clip](https://github.com/mlfoundations/open_clip) 到 `/root/Documents/DEMOS`

```bash
conda activate CLIP

# 从本地安装 open_clip
# 安装的包（/root/Documents/DEMOS/open_clip/src）将被链接到本地环境中，而不是被复制到环境中
# 这使得开发人员可以方便地进行包的开发和测试，并且对包所做的更改会立即反映在本地环境中
cd /root/Documents/DEMOS/open_clip
make install 

# 安装 open_clip/src/training 所需的依赖
make install-training
```

- 参考以下 Makefile 内容
    
    ```makefile
    install: ## [Local development] Upgrade pip, install requirements, install package.
    	python -m pip install -U pip
    	python -m pip install -e .
    
    install-training:
    	python -m pip install -r requirements-training.txt
    ```
    
- 以及自动创建的链接文件 open-clip-torch.egg-link
    
    ```makefile
    # /opt/homebrew/Caskroom/miniforge/base/envs/CLIP/lib/python3.8/site-packages/open-clip-torch.egg-link
    /Users/x1a/Documents/Python/Demo/open_clip/src
    ../
    ```
    
- 测试
    
    ```bash
    import torch
    torch.__version__
    # '1.13.1'
    torch.cuda.is_available()
    # True
    
    import open_clip
    open_clip.list_pretrained()
    ```
    

# 安装 fairseq 依赖

```python
cd ~/Documents/DEMOS
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
# 自动将 torch 更新为了 '2.0.0+cu117'
# 数据已经下好，也就无所谓 clip-benchmark 了

pip install sacremoses
pip install fastBPE
```