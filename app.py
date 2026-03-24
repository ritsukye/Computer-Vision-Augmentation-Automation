import streamlit as st
from PIL import Image

# Configure the Streamlit page with a wide layout for side-by-side image display
st.set_page_config(page_title="Image Augmentation Tool", layout="wide")
st.title("Image Augmentation Tool")

# File uploader accepting common image formats
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp", "webp"])

# --- Test Case Definitions ---
# Each category contains a list of test cases with an id, name, and description.
# TODO: When implementing augmentations, create a function for each test case in a
# separate augmentations.py module. Each function should accept a PIL Image and return
# the augmented PIL Image. Map each test case id to its corresponding function.
TEST_CASES = {
    # Season Change: These require advanced color manipulation and potentially
    # AI-based style transfer. Consider using color remapping for simple cases
    # or a pre-trained model (e.g. CycleGAN) for realistic season transforms.
    # TODO: Implement season augmentations — start with simple color/tone shifts
    # using PIL (hue rotation, color balance), then explore neural style transfer
    # for more realistic results.
    "Season Change": [
        {"id": "S1", "name": "Summer → Winter", "description": "Convert a summer scene to a winter scene with snow overlay and cold color tones"},
        {"id": "S2", "name": "Summer → Autumn", "description": "Shift green foliage to orange/red autumn palette"},
        {"id": "S3", "name": "Summer → Spring", "description": "Add bloom effects, brighter greens, and warmer lighting"},
        {"id": "S4", "name": "Winter → Summer", "description": "Remove snow, add green vegetation and warm tones"},
        {"id": "S5", "name": "Day → Night", "description": "Darken scene, add artificial lighting and blue tint"},
    ],

    # Flip: Simple geometric transforms using PIL's Image.transpose().
    # TODO: Implement using Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM,
    # and a combination of both.
    "Flip": [
        {"id": "F1", "name": "Horizontal Flip", "description": "Mirror the image along the vertical axis (left ↔ right)"},
        {"id": "F2", "name": "Vertical Flip", "description": "Mirror the image along the horizontal axis (top ↔ bottom)"},
        {"id": "F3", "name": "Both Axes Flip", "description": "Flip horizontally and vertically (equivalent to 180° rotation)"},
    ],

    # Rotation: Use PIL's Image.rotate() with expand=True to avoid cropping.
    # TODO: Implement fixed rotations (90, 180, 270) using Image.transpose() for
    # lossless rotation. For arbitrary angles, use Image.rotate() with a fill color
    # or transparency for the exposed background.
    "Rotation": [
        {"id": "R1", "name": "Rotate 90°", "description": "Rotate image 90° clockwise"},
        {"id": "R2", "name": "Rotate 180°", "description": "Rotate image 180°"},
        {"id": "R3", "name": "Rotate 270°", "description": "Rotate image 270° clockwise (90° counter-clockwise)"},
        {"id": "R4", "name": "Arbitrary Rotation", "description": "Rotate by a user-specified angle (e.g. 15°, 45°) with background fill"},
        {"id": "R5", "name": "Small Tilt (±5°)", "description": "Slight random tilt to simulate camera misalignment"},
    ],

    # Resolution: Use PIL's Image.resize() with different resampling filters.
    # TODO: Implement using Image.resize() with Image.LANCZOS for downscaling
    # and Image.BICUBIC for upscaling. For the downscale+upscale artifact test,
    # chain a downscale then upscale to simulate JPEG-like quality loss.
    "Resolution": [
        {"id": "RE1", "name": "Downscale 50%", "description": "Reduce resolution to half (simulate low-quality input)"},
        {"id": "RE2", "name": "Downscale 25%", "description": "Reduce resolution to quarter (heavy quality loss)"},
        {"id": "RE3", "name": "Upscale 2×", "description": "Double the resolution using interpolation"},
        {"id": "RE4", "name": "Downscale + Upscale", "description": "Reduce then enlarge to simulate compression artifacts"},
        {"id": "RE5", "name": "Change Aspect Ratio", "description": "Stretch or squash to a different aspect ratio (e.g. 4:3 → 16:9)"},
    ],
}

# --- Image Upload and Side-by-Side Display ---
# When an image is uploaded, show the original and augmented versions side by side.
# TODO: Once augmentation functions are implemented, apply the user-selected
# augmentation to produce the augmented image instead of showing the original twice.
# Add a selectbox or radio button to let the user pick which test case to apply.
if uploaded_file:
    image = Image.open(uploaded_file)

    # Two-column layout: original on the left, augmented result on the right
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(image, use_container_width=True)
    with col2:
        # TODO: Replace `image` with the augmented output once augmentation is wired up
        st.subheader("Augmented")
        st.image(image, use_container_width=True)

    st.divider()

# --- Display Test Cases as Cards ---
# Renders each category of test cases in a columnar grid so the user can see
# all available augmentation scenarios at a glance.
# TODO: Make each test case clickable — when selected, apply that augmentation
# to the uploaded image and display the result in the "Augmented" column above.
# Update the status field to reflect implementation progress (e.g. "Implemented",
# "In Progress", "Not Implemented").
st.header("Augmentation Test Cases")

for category, cases in TEST_CASES.items():
    st.subheader(category)
    # Create one column per test case in this category
    cols = st.columns(len(cases))
    for i, tc in enumerate(cases):
        with cols[i]:
            # TODO: Change status dynamically based on whether the augmentation
            # function exists and is wired up
            status = "🔲 Not Implemented"
            st.markdown(
                f"**{tc['id']}: {tc['name']}**\n\n"
                f"{tc['description']}\n\n"
                f"Status: {status}"
            )

# Prompt the user to upload an image if none is provided yet
if not uploaded_file:
    st.info("Upload an image above to preview augmentations.")
