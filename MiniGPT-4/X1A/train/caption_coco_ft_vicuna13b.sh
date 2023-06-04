cd /root/Documents/DEMOS/MiniGPT-4
nohup /root/anaconda3/envs/minigpt4/bin/python -m torch.distributed.run --nproc_per_node=1 train.py --cfg-path /root/Documents/DEMOS/MiniGPT-4/X1A/train/caption_coco_ft_vicuna13b.yaml >/root/Documents/DEMOS/MiniGPT-4/X1A/output/train/caption_coco_ft_vicuna13b.log& 2>&1
tail -f /root/Documents/DEMOS/MiniGPT-4/X1A/output/train/caption_coco_ft_vicuna13b.log