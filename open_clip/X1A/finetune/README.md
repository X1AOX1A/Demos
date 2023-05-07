# Fine-tune CoCa on mscoco

To fine-tune coca on mscoco, first create the dataset, one way is using a csvdataset and perhaps the simplest way to do it is using [CLIP_benchmark](https://github.com/LAION-AI/CLIP_benchmark) which in turn uses [pycocotools](https://github.com/cocodataset/cocoapi) (that can be used also by itself).