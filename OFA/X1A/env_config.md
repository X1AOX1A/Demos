# OFA

```python
conda create -n OFA python=3.8
conda activate OFA

cd ~/Documents/DEMOS
git clone https://github.com/OFA-Sys/OFA
cd OFA
pip install -r requirements.txt
pip install tensorboard
pip install torchaudio
pip install g2p_en
pip install sacremoses

# https://stackoverflow.com/questions/61320572/modulenotfounderror-no-module-named-tensorboard
pip install protobuf==3.20.*
```

- `libGL.so.1: cannot open shared object file: No such file or directory`
    
    ```python
    Exception has occurred: ImportError
    libGL.so.1: cannot open shared object file: No such file or directory
      File "/root/Documents/DEMOS/OFA/utils/vision_helper.py", line 6, in <module>
        import cv2
      File "/root/Documents/DEMOS/OFA/data/cv_data/image_classify_dataset.py", line 17, in <module>
        from utils.vision_helper import RandomAugment
      File "/root/Documents/DEMOS/OFA/tasks/cv_tasks/image_classify.py", line 19, in <module>
        from data.cv_data.image_classify_dataset import ImageClassifyDataset
      File "/root/Documents/DEMOS/OFA/tasks/cv_tasks/__init__.py", line 1, in <module>
        from .image_classify import ImageClassifyTask
      File "/root/Documents/DEMOS/OFA/tasks/__init__.py", line 1, in <module>
        from .cv_tasks import *
      File "/root/Documents/DEMOS/OFA/utils/eval_utils.py", line 17, in <module>
        from tasks.nlg_tasks.gigaword import fix_tokenization
      File "/root/Documents/DEMOS/OFA/X1A/inference/caption_infer.py", line 8, in <module>
        from utils.eval_utils import eval_step
    ImportError: libGL.so.1: cannot open shared object file: No such file or directory
    ```
    

```python
apt install libgl1-mesa-glx
```

[ImportError: libGL.so.1: cannot open shared object file: No such file or directory](https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo#comment132094077_55313610)