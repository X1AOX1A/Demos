conda activate minigpt4
cd /root/Documents/DEMOS/MiniGPT-4
nohup /root/anaconda3/envs/lavis/bin/python -m torch.distributed.run --nproc_per_node=4 evaluate.py --cfg-path /root/Documents/DEMOS/MiniGPT-4/X1A/eval/caption_coco_zeroshot_vicuna13b_eval.yaml >/root/Documents/DEMOS/MiniGPT-4/X1A/eval/caption_coco_zeroshot_vicuna13b_eval.log& 2>&1
tail -f /root/Documents/DEMOS/MiniGPT-4/X1A/eval/caption_coco_zeroshot_vicuna13b_eval.log