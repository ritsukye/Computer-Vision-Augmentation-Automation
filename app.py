import albumentations as A
import streamlit as st
import numpy as np
import cv2
import yaml
import io
import zipfile
from pathlib import Path
from PIL import Image
from augment.image_transforms import build_aug

# ****IMPORTANT NOTE: the GUI is preview only and one audio clip/image at a time,
# and never touches the disk. it exists so someone can preview what a transformation
# looks like before deciding to generate. it remains local via run.py so hosted intance
# remains small for a larger classrom use

st.set_page_config(page_title="Augmentation Test Cases", layout="wide")
st.title("Augmentation Test Case Generator")

MODE = st.radio("Mode", ["Image", "Audio"], horizontal=True)
st.divider()

# load augmentation config
BASE_DIR = Path(__file__).resolve().parent

@st.cache_data
def load_cfg():
    cfg_path = BASE_DIR / "augment" / "image_config.yaml"
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

CFG = load_cfg()





TRANSFORM_GROUPS = ["appearance", "obscure", "geometry"]

def get_all_augmentations():
    """returns list of (display_name, group, aug_name, aug_cfg) for enabled transforms"""

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

# extra gui not in yaml // alreadyfixed points
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
    shift = A.RGBShift(r_shift_limit=(20, 20) if warm else (-20, -20), g_shift_limit=(5, 5) if warm else (-5, -5),
        b_shift_limit=(-10, -10) if warm else (15, 15), p=1)
    return A.Compose([shift])(image=img)["image"]

def _autumn(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + 15) % 180  # shift hue toward orange
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.3, 0, 255)  # boost saturation
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def apply_aug(name, tcfg, img):
    aug = build_aug(name, tcfg)
    if aug is None:
        return None
    return A.Compose([aug])(image=img)["image"]


if MODE == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp", "tiff", "webp"])

    if not uploaded_file:
        st.info("Upload an image above to generate augmentation test cases.")
        st.stop()

    # need to read image as BGR numpy array for ablumentations
    file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    st.image(img_rgb, caption="Original Image", width=400)
    st.divider()

    # using tab 1 for the YAML-defined augmentations, tab 2 for extra test cases
    tab1, tab2 = st.tabs(["Config Augmentations", "Additional Test Cases"])

    all_results = {}

    # ======= TAB 1
    with tab1:
        st.markdown("These augmentations are defined in the 'image_config.yaml' file and are used by the batch pipeline (run.py).")


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

    # ======= TAB 2
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

    # download section
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

# audio ui
else:
    st.info(
        "Applies every enabled transform in 'audio_config.yaml' to one uploaded clip. "
        "Bulk generation over a dataset stays local via 'run.py'"
    )

    uploaded_audio = st.file_uploader("Upload an audio clip", type=["wav", "mp3", "flac", "ogg", "m4a"])

    if not uploaded_audio:
        st.stop()

    st.audio(uploaded_audio)
    st.divider()

    import tempfile
    import soundfile as sf
    import librosa
    from augment.audio_transforms import build_aug as build_audio_aug

    @st.cache_data
    def load_audio_cfg():
        cfg_path = BASE_DIR/"augment"/"audio_config.yaml"
        with open(cfg_path, "r") as f:
            return yaml.safe_load(f)

    audio_cfg = load_audio_cfg()
    audio_groups = ["levels", "noise", "temporal", "spectral", "codec"]

    # load the uploaded clip to a float32 mono array
    suffix = Path(uploaded_audio.name).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_audio.getvalue())
        tmp_path = tmp.name
    try:
        samples, sr = librosa.load(tmp_path, sr=None, mono=True)
    except Exception as e:
        st.error(f"Could not decode this clip ({e})! Try a WAV/FLAC file.")
        st.stop()

    def to_wav_bytes(data, rate):
        buf = io.BytesIO()
        sf.write(buf, data, rate, format="WAV")
        return buf.getvalue()

    for group in audio_groups:
        if group not in audio_cfg:
            continue
        enabled = [(n, c) for n, c in audio_cfg[group].items() if c.get("enabled")]
        if not enabled:
            continue
        st.subheader(group.title())
        for name, tcfg in enabled:
            label = name.replace("_", " ").title()
            aug = build_audio_aug(name, tcfg)
            if aug is None:
                st.warning(f"{label}: transform not available")
                continue
            try:
                out = aug(samples=samples, sample_rate=sr)
                st.caption(f"IC_{name}")
                st.audio(to_wav_bytes(out, sr), format="audio/wav")
            except Exception as e:
                st.error(f"{label}: {e}")
