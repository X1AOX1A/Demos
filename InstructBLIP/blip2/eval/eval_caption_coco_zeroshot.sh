conda activate lavis
cd /root/Documents/DEMOS/LAVIS
nohup /root/anaconda3/envs/lavis/bin/python -m torch.distributed.run --nproc_per_node=4 evaluate.py --cfg-path /root/Documents/DEMOS/InstructBLIP/blip2/eval/caption_coco_zeroshot_opt2.7b_eval.yaml >/root/Documents/DEMOS/InstructBLIP/blip2/eval/eval_caption_coco_zeroshot.log& 2>&1
tail -f /root/Documents/DEMOS/InstructBLIP/blip2/eval/eval_caption_coco_zeroshot.log