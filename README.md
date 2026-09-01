# Image and Audio Augmentation for Quality Assurance Testing Automation

This is a Python-based application for automating augmentation of various data types (image and audio), as part of an extension of a research project under the supervision of Professor Jerry Gao. Current features include transformations of image appearance, obscurity, and geometry, alongside various transformations of audio.

The batch pipeline will remain local, so the result will be generated with local CPU/RAM. Hosted Streamlit app is an optional feature that allows the user to upload ONE image/clip for preview purposes. In this way, users can understand what exact transformations are and selectively generate the appropriate augmentations they would like to test the app with.

## Installation

Use the package manager pip in the terminal of the project to install the requirements with: pip install -r requirements.txt

```bash
pip install -r requirements.txt
```

# Instructions for Augmenting Images

ACCEPTED IMAGE FILE TYPES: ".jpg", ".jpeg", ".png", ".bmp", ".tiff"

Place your preferred .zip folder of images under the project's uploads/ folder. Be sure to name it "image_dataset.zip"

Run the augmentation by typing in the project's terminal:
```bash
python run.py
```

The augmented images can be found under data/augmented/image/. There should 21 generated test case **types** of the .zip file uploaded, meaning if 10 images are uploaded, 210 augmented test cases are created.

# Instructions for Augmenting Audio (WIP)

Audio support is being worked in progress but follows the same local process as images:

- `augment/audio_config.yaml` — transform config, mirrors image_config.yaml's structure (curr all transforms disabled)
- `augment/audio_transforms.py` — `build_aug(name, cfg)` not yet wired to a real audio library
- `augment/augment_runner.py` — `process_audio()` already unzips `uploads/audio_dataset.zip` into `data/raw/audio/` and will write results to `data/augmented/audio/IC_<name>/` once transforms are implemented
- The GUI's Audio mode lets you upload and preview a clip and lists the transforms queued up in `audio_config.yaml`

Image and audio data are kept in separate `image/`/`audio/` subfolders under `data/raw/` and `data/augmented/` (and separate `image_dataset.zip` / `audio_dataset.zip` uploads) so the two modalities can't collide on folder or file names, even where transform names might otherwise overlap (e.g. "noise", "shift").

## GUI

A Streamlit-based GUI is available for interactively uploading a single image and previewing all augmentation test cases.

Run the GUI with:
```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

Use the **Image / Audio** switch at the top to choose a modality. In Image mode the GUI has two tabs:
- **Config Augmentations**: All appearance, obscure, and geometry transforms defined in `augment/image_config.yaml`
- **Additional Test Cases**: Flip variants, fixed rotations, resolution changes, and season/lighting effects

Upload an image to see every augmentation applied. Use the **Download as ZIP** button to export all generated test case images at once (only augmentations of singular uploads).
