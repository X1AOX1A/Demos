import os
import pandas as pd
from clip_benchmark.datasets.builder import build_dataset

DATA_DIR = "/root/Documents/DATASETS/MS_COCO"

ds = build_dataset("mscoco_captions", root=DATA_DIR, split="train") # this downloads the dataset if it is not there already
coco = ds.coco
imgs = coco.loadImgs(coco.getImgIds())
future_df = {"filepath":[], "title":[]}
for img in imgs:
    caps = coco.imgToAnns[img["id"]]
    for cap in caps:
        future_df["filepath"].append(
            os.path.join(DATA_DIR, "train2014", img["file_name"]))
        future_df["title"].append(cap["caption"])
pd.DataFrame.from_dict(future_df).to_csv(
  os.path.join(DATA_DIR, "train2014.csv"), index=False, sep="\t"
)