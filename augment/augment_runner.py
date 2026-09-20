import zipfile
import yaml
import albumentations as A
import librosa
import soundfile as sf
import random
import cv2
import numpy as np
from pathlib import Path
from augment.image_transforms import build_aug as build_image_aug
from augment.audio_transforms import build_aug as build_audio_aug

# ================= accepted file formats  for audio and image
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ================= paths
UPLOAD_DIR = PROJECT_ROOT/"uploads"
IMAGE_UPLOAD_ZIP = UPLOAD_DIR/"image_dataset.zip"
AUDIO_UPLOAD_ZIP = UPLOAD_DIR/"audio_dataset.zip"

RAW_ROOT = PROJECT_ROOT/"data"/"raw"
OUT_ROOT = PROJECT_ROOT/"data"/"augmented"
IMAGE_RAW_ROOT = RAW_ROOT/"image"
IMAGE_OUT_ROOT = OUT_ROOT/"image"
AUDIO_RAW_ROOT = RAW_ROOT/"audio"
AUDIO_OUT_ROOT = OUT_ROOT/"audio"

BASE_DIR = Path(__file__).resolve().parent

# ================= transform groups for image and audio from the yamls
IMAGE_TRANSFORM_GROUPS = ["appearance", "obscure", "geometry"]
AUDIO_TRANSFORM_GROUPS = ["levels", "noise", "temporal", "spectral", "codec"]

# ================= extraction and loading prior to augmentattion
# extract a dataset zip into a raw root
def unzip_dataset(zip_path, raw_root):
    raw_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(raw_root)

# return files under root matching the given extensions
def find_files(root, exts):
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]


# load modality's yaml config (image_config.yaml/audio_config.yaml)
def load_cfg(filename):
    cfg_path = BASE_DIR / filename
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found at: {cfg_path}")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

def seed_from_cfg(cfg):
    if cfg.get("deterministic", False):
        seed = cfg.get("seed", 42)
        np.random.seed(seed)
        random.seed(seed)

# ================= augmentation for images
# run image augmentation over data/raw/image to data/augmented/image/IC_<name>/..
def process_images():
    if not IMAGE_UPLOAD_ZIP.exists():
        return

    unzip_dataset(IMAGE_UPLOAD_ZIP, IMAGE_RAW_ROOT)
    cfg = load_cfg("image_config.yaml")
    seed_from_cfg(cfg)

    for img_path in find_files(IMAGE_RAW_ROOT, IMAGE_EXTS):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        rel = img_path.parent.relative_to(IMAGE_RAW_ROOT)

        for group in IMAGE_TRANSFORM_GROUPS:
            if group not in cfg:
                continue

            for name, tcfg in cfg[group].items():
                if not tcfg.get("enabled"):
                    continue

                aug = build_image_aug(name, tcfg)
                if aug is None:
                    continue


                try:
                    out = A.Compose([aug])(image=img)["image"]
                except Exception:
                    continue

                out_dir = IMAGE_OUT_ROOT / f"IC_{name}" / rel
                out_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_dir / img_path.name), out)

# ================= augmentation for audio
# audio augmentation over data/raw/audio to data/augmented/audio/IC_<name>/...
def process_audio():
    if not AUDIO_UPLOAD_ZIP.exists():
        return

    unzip_dataset(AUDIO_UPLOAD_ZIP, AUDIO_RAW_ROOT)
    cfg = load_cfg("audio_config.yaml")
    seed_from_cfg(cfg)

    for audio_path in find_files(AUDIO_RAW_ROOT, AUDIO_EXTS):
        try:
            samples, sr = librosa.load(str(audio_path), sr=None, mono=False)
        except Exception:
            continue

        rel = audio_path.parent.relative_to(AUDIO_RAW_ROOT)

        for group in AUDIO_TRANSFORM_GROUPS:
            if group not in cfg:
                continue

            for name, tcfg in cfg[group].items():
                if not tcfg.get("enabled"):
                    continue

                aug = build_audio_aug(name, tcfg)
                if aug is None:
                    continue

                try:
                    out = aug(samples=samples, sample_rate=sr)
                except Exception:
                    continue

                out_dir = AUDIO_OUT_ROOT / f"IC_{name}" / rel
                out_dir.mkdir(parents=True, exist_ok=True)
                data = out.T if out.ndim == 2 else out
                sf.write(str(out_dir / f"{audio_path.stem}.wav"), data, sr)


def run():
    process_images()
    process_audio()

if __name__ == "__main__":
    run()
