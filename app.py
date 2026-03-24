import streamlit as st
import numpy as np
import cv2
import yaml
import io
import zipfile
from pathlib import Path
from PIL import Image
import albumentations as A
from augment.transforms import build_aug

st.set_page_config(page_title="Image Augmentation Test Cases", layout="wide")
st.title("Image Augmentation Test Case Generator")

# ---------------------------------------------------------------------------
# Load augmentation config
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

@st.cache_data
def load_cfg():
    cfg_path = BASE_DIR / "augment" / "aug_config.yaml"
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

CFG = load_cfg()

# ---------------------------------------------------------------------------
# Gather every enabled augmentation from the YAML config
# ---------------------------------------------------------------------------
TRANSFORM_GROUPS = ["appearance", "obscure", "geometry"]

def get_all_augmentations():
    """Return list of (display_name, group, aug_name, aug_cfg) for enabled transforms."""
    augs = []
    labels = {
        "hue_shift": "Hue Shift",
        "saturation_shift": "Saturation Shift",
        "brightness_shift": "Brightness Shift",
        "contrast_shift": "Contrast Shift",
        "gamma_shift": "Gamma Shift",
        "rgb_shift": "RGB Shift (Channel Noise)",
        "clahe": "CLAHE (Local Contrast)",
        "white_balance_shift": "White Balance Shift",
        "gaussian_blur": "Gaussian Blur",
        "motion_blur": "Motion Blur",
        "gaussian_noise": "Gaussian Noise",
        "jpeg_compress": "JPEG Compression Artifacts",
        "cutout": "Random Erasing (Cutout)",
        "resize_degrade": "Downscale Degradation",
        "rotate": "Rotation",
        "crop": "Random Crop & Resize",
        "zoom": "Zoom",
        "shift": "Shift Translation",
        "shear": "Shear",
        "perspective": "Perspective Transform",
        "flip": "Horizontal Flip",
    }
    for group in TRANSFORM_GROUPS:
        if group not in CFG:
            continue
        for name, tcfg in CFG[group].items():
            if not tcfg.get("enabled"):
                continue
            display = labels.get(name, name.replace("_", " ").title())
            augs.append((display, group.title(), name, tcfg))
    return augs

ALL_AUGS = get_all_augmentations()

# ---------------------------------------------------------------------------
# Extra GUI-only test cases (not in the YAML config)
# ---------------------------------------------------------------------------
EXTRA_CASES = {
    "Flip": [
        ("Horizontal Flip", lambda img: A.Compose([A.HorizontalFlip(p=1)])(image=img)["image"]),
        ("Vertical Flip", lambda img: A.Compose([A.VerticalFlip(p=1)])(image=img)["image"]),
        ("Both Axes Flip", lambda img: A.Compose([A.HorizontalFlip(p=1), A.VerticalFlip(p=1)])(image=img)["image"]),
    ],
    "Rotation": [
        ("Rotate 90\u00b0", lambda img: cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)),
        ("Rotate 180\u00b0", lambda img: cv2.rotate(img, cv2.ROTATE_180)),
        ("Rotate 270\u00b0", lambda img: cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)),
        ("Small Tilt (\u00b15\u00b0)", lambda img: A.Compose([A.Rotate(limit=5, p=1)])(image=img)["image"]),
    ],
    "Resolution": [
        ("Downscale 50%", lambda img: _rescale(img, 0.5)),
        ("Downscale 25%", lambda img: _rescale(img, 0.25)),
        ("Upscale 2\u00d7", lambda img: _rescale(img, 2.0)),
        ("Downscale + Upscale", lambda img: _rescale(_rescale(img, 0.25), 4.0)),
        ("Stretch 16:9", lambda img: cv2.resize(img, (int(img.shape[1] * 1.33), int(img.shape[0] * 0.75)))),
    ],
    "Season / Lighting": [
        ("Day \u2192 Night", lambda img: _day_to_night(img)),
        ("Warm Tone (Summer)", lambda img: _color_temp(img, warm=True)),
        ("Cool Tone (Winter)", lambda img: _color_temp(img, warm=False)),
        ("Autumn Palette", lambda img: _autumn(img)),
    ],
}

def _rescale(img, factor):
    h, w = img.shape[:2]
    new_w, new_h = max(1, int(w * factor)), max(1, int(h * factor))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4 if factor < 1 else cv2.INTER_CUBIC)

def _day_to_night(img):
    dark = (img * 0.3).astype(np.uint8)
    blue_tint = np.full_like(img, (50, 0, 0), dtype=np.uint8)  # BGR blue
    return cv2.add(dark, blue_tint)

def _color_temp(img, warm=True):
    shift = A.RGBShift(r_shift_limit=(20, 20) if warm else (-20, -20),
                       g_shift_limit=(5, 5) if warm else (-5, -5),
                       b_shift_limit=(-10, -10) if warm else (15, 15), p=1)
    return A.Compose([shift])(image=img)["image"]

def _autumn(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + 15) % 180  # shift hue toward orange
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.3, 0, 255)  # boost saturation
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

# ---------------------------------------------------------------------------
# Helper: apply a config-based augmentation
# ---------------------------------------------------------------------------
def apply_aug(name, tcfg, img):
    aug = build_aug(name, tcfg)
    if aug is None:
        return None
    return A.Compose([aug])(image=img)["image"]

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp", "tiff", "webp"])

if not uploaded_file:
    st.info("Upload an image above to generate augmentation test cases.")
    st.stop()

# Read image as BGR numpy array (what OpenCV / albumentations expect)
file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

st.image(img_rgb, caption="Original Image", width=400)
st.divider()

# ---------------------------------------------------------------------------
# Tab 1: Config-based augmentations   Tab 2: Extra test cases
# ---------------------------------------------------------------------------
tab1, tab2 = st.tabs(["Config Augmentations", "Additional Test Cases"])

# Collect all generated images for the zip download
all_results = {}  # name -> RGB numpy array

# --- Tab 1: augmentations from aug_config.yaml ---
with tab1:
    st.markdown("These augmentations are defined in `aug_config.yaml` and used by the batch pipeline (`run.py`).")
    for group in TRANSFORM_GROUPS:
        group_augs = [(d, g, n, c) for d, g, n, c in ALL_AUGS if g == group.title()]
        if not group_augs:
            continue
        st.subheader(group.title())
        cols = st.columns(min(len(group_augs), 4))
        for idx, (display, _, name, tcfg) in enumerate(group_augs):
            col = cols[idx % len(cols)]
            with col:
                result = apply_aug(name, tcfg, img_bgr)
                if result is not None:
                    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                    st.image(result_rgb, caption=display, use_container_width=True)
                    all_results[f"IC_{name}"] = result_rgb
                else:
                    st.warning(f"{display}: transform not available")

# --- Tab 2: extra test cases (flip, rotation, resolution, season) ---
with tab2:
    for category, cases in EXTRA_CASES.items():
        st.subheader(category)
        cols = st.columns(min(len(cases), 4))
        for idx, (label, func) in enumerate(cases):
            col = cols[idx % len(cols)]
            with col:
                try:
                    result = func(img_bgr)
                    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                    st.image(result_rgb, caption=label, use_container_width=True)
                    safe_name = label.replace(" ", "_").replace("/", "-")
                    all_results[f"TC_{safe_name}"] = result_rgb
                except Exception as e:
                    st.error(f"{label}: {e}")

st.divider()

# ---------------------------------------------------------------------------
# Download all augmented images as a zip
# ---------------------------------------------------------------------------
st.subheader("Download All Test Cases")
total = len(all_results)
st.write(f"**{total}** augmented test case images ready.")

if st.button("Download as ZIP"):
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, rgb_img in all_results.items():
            img_pil = Image.fromarray(rgb_img)
            img_bytes = io.BytesIO()
            img_pil.save(img_bytes, format="PNG")
            zf.writestr(f"{name}.png", img_bytes.getvalue())
    st.download_button(
        label=f"Download {total} images (ZIP)",
        data=zip_buf.getvalue(),
        file_name="augmented_test_cases.zip",
        mime="application/zip",
    )
