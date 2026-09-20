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

# Instructions for Augmenting Audio

ACCEPTED AUDIO FILE TYPES: ".wav", ".mp3", ".flac", ".ogg", ".m4a"

Place your preferred .zip folder of clips under the project's uploads/ folder. Be sure to name it "audio_dataset.zip", then run `python run.py` (the same command handles both modalities).

The augmented clips can be found under `data/augmented/audio/`, always written as `.wav` regardless of input format. There are 12 generated test case **types**, so 10 uploaded clips produce 120 augmented test cases.

Audio transforms are backed by [`audiomentations`](https://iver56.github.io/audiomentations/) and grouped in `augment/audio_config.yaml` with transformation groups such as levels, noise, temporal, spectral, and codec.

Toggle transforms with `enabled: true/false` and tune their parameters in that file, exactly like `image_config.yaml`. The GUI's Audio mode applies every enabled transform to one uploaded clip for preview.

Image and audio data are kept in separate `image/`/`audio/` subfolders under `data/raw/` and `data/augmented/` (and separate `image_dataset.zip` / `audio_dataset.zip` uploads) so the two modes can't collide on folder or file names, even where transform names might otherwise overlap (e.g. "noise", "shift").

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

In Audio mode, upload a single clip to hear every enabled transform from `augment/audio_config.yaml` applied in place.
