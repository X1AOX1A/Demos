conda activate lavis
cd /root/Documents/DEMOS/LAVIS
nohup /root/anaconda3/envs/lavis/bin/python -m torch.distributed.run --nproc_per_node=4 train.py --cfg-path /root/Documents/DEMOS/LAVIS/X1A/blip2/train/caption_coco_ft.yaml >/root/Documents/DEMOS/LAVIS/X1A/blip2/train/train_caption_coco.log& 2>&1
tail -f /root/Documents/DEMOS/LAVIS/X1A/blip2/train/train_caption_coco.log