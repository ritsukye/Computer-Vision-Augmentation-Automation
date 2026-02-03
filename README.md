# Image and Audio Augmentation for Quality Assurance Testing Automation

This is a Python-based application for automating augmentation of various data types, as part of an extension of a research project under the supervision of Professor Jerry Gao. Current features include transformations of image appearance, obscurity, and geometry.

## Installation

Use the package manager pip in the terminal of the project to install the requirements with: pip install -r requirements.txt

```bash
pip install -r requirements.txt
```

# Instructions for Augmenting Images

ACCEPTED IMAGE FILE TYPES: ".jpg", ".jpeg", ".png", ".bmp", ".tiff"

Place your preferred .zip folder of images under the project's uploads/ folder. Be sure to name it "dataset.zip" 

Run the augmentation by typing in the project's terminal:
```bash
python run.py
```

The augmented images can be found under the data/augmented/ folder. There should 21 generated test case **types** of the .zip file uploaded, meaning if 10 images are uploaded, 210 augmented test cases are created.


