import zipfile
import yaml

import cv2
import numpy as np
from pathlib import Path
import albumentations as A
import random
from augment.transforms import build_aug

# we accept these image formats
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

# project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# paths
UPLOAD_ZIP = PROJECT_ROOT / "uploads" / "dataset.zip"
RAW_ROOT = PROJECT_ROOT / "data" / "raw"
OUT_ROOT = PROJECT_ROOT / "data" / "augmented"
BASE_DIR = Path(__file__).resolve().parent

# extract images from dataset
def unzip_dataset():
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(UPLOAD_ZIP, "r") as z:
        z.extractall(RAW_ROOT)

# return images
def find_images(root):
    return [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS]

# load config file with .yaml formatted augmentations
def load_cfg():
    cfg_path = Path(__file__).resolve().parent / "aug_config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found at: {cfg_path}")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

# run the augmentation process, called in run.py
def run():
    unzip_dataset()
    cfg = load_cfg()

    if cfg.get("deterministic", False):
        seed = cfg.get("seed", 42)
        np.random.seed(seed)
        random.seed(seed)

    for img_path in find_images(RAW_ROOT):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        rel = img_path.parent.relative_to(RAW_ROOT)

        #only process these specific groups, weather / season will come next
        transform_groups = ["appearance", "obscure", "geometry"]

        for group in transform_groups:
            if group not in cfg:
                continue

            transforms = cfg[group]
            for name, tcfg in transforms.items():

                if not tcfg.get("enabled"):
                    continue

                aug = build_aug(name, tcfg)
                if aug is None:
                    continue

                try:
                    composed = A.Compose([aug])
                    out = composed(image=img)["image"]
                except Exception:
                    continue

                ic_name = f"IC_{name}"
                out_dir = OUT_ROOT / ic_name / rel
                out_dir.mkdir(parents=True, exist_ok=True)

                cv2.imwrite(str(out_dir / img_path.name), out)

# guard run() to only run as a script, not as a library
if __name__ == "__main__":
    run()
