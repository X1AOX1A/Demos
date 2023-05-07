DATA_DIR=/root/Documents/DATASETS/MS_COCO

export CUDA_VISIBLE_DEVICES="1,2,3"

torchrun --nproc_per_node 3 -m training.main \
    --dataset-type "csv" \
    --train-data "${DATA_DIR}/train2014.csv" \
    --warmup 1000 \
    --batch-size 40 \
    --lr 1e-5 \
    --wd 0.1 \
    --epochs 1 \
    --workers 3 \
    --model "coca_ViT-L-14" \
    --coca-contrastive-loss-weight 0 \
    --coca-caption-loss-weight 1 \
    --log-every-n-steps 100
    # --report-to "wandb" \

# The only relevant change compared to pre-training are the two arguments
# --coca-contrastive-loss-weight 0
# --coca-caption-loss-weight 1
# which make the model only train the generative side.