
import streamlit as st
import cv2
import numpy as np
from src.models import perform_custom_segmentation
from src.utils import resize_image, download_image
import os
import torch

# Constants
TARGET_SIZE = (750, 750)

def get_parameters_from_sidebar() -> dict:
    """Get segmentation parameters from sidebar"""
    st.sidebar.header("Segmentation Parameters")
    param_names = ['train_epoch', 'mod_dim1', 'mod_dim2', 'min_label_num', 'max_label_num']
    param_values = [(1, 200, 43), (1, 128, 67), (1, 128, 63), (1, 20, 3), (1, 200, 25)]
    params = {name: st.sidebar.slider(name.replace('_', ' ').title(), *values) for name, values in zip(param_names, param_values)}
    
    # Add sliders for target size width and height
    target_size_width = st.sidebar.number_input("Target Size Width", 100, 1200, 750)
    target_size_height = st.sidebar.number_input("Target Size Height", 100, 1200, 750)
    params['target_size'] = (target_size_width, target_size_height)
    
    return params
def display_segmentation_results() -> None:
    """Display segmentation results"""
    st.image(st.session_state.segmented_image, caption='Updated Segmented Image', use_column_width=True)

def randomize_colors() -> None:
    """Randomize colors for segmentation labels"""
    unique_labels = np.unique(st.session_state.segmented_image.reshape(-1, 3), axis=0)
    random_colors = {tuple(label): tuple(np.random.randint(0, 256, size=3)) for label in unique_labels}

    for old_color, new_color in random_colors.items():
        mask = np.all(st.session_state.segmented_image == np.array(old_color), axis=-1)
        st.session_state.segmented_image[mask] = new_color

    # Update color mappings in session state
    st.session_state.new_colors.update(random_colors)
    st.session_state.image_update_trigger += 1  # Trigger image update

def handle_color_picking() -> None:
    """Handle color picking and other functionalities"""
    unique_labels = np.unique(st.session_state.segmented_image.reshape(-1, 3), axis=0)
    for i, label in enumerate(unique_labels):
        hex_label = f'#{label[0]:02x}{label[1]:02x}{label[2]:02x}'
        new_color = st.color_picker(f"Choose a new color for label {i}", value=hex_label, key=f"label_{i}")
        new_color_rgb = tuple(int(new_color.lstrip('#')[j:j+2], 16) for j in (0, 2, 4))
        st.session_state.new_colors[tuple(label)] = new_color_rgb

    # Convert the new colors to hexadecimal for comparison
    new_colors_hex = {tuple(label): f'#{label[0]:02x}{label[1]:02x}{label[2]:02x}' for label in st.session_state.new_colors.values()}

    for old_color, new_color in st.session_state.new_colors.items():
        # Convert the old color to hexadecimal for comparison
        old_color_hex = f'#{old_color[0]:02x}{old_color[1]:02x}{old_color[2]:02x}'
        # Find the corresponding new color in hexadecimal
        new_color_hex = new_colors_hex[new_color]
        # Update the segmented image with the new color
        mask = np.all(st.session_state.segmented_image == np.array(old_color), axis=-1)
        st.session_state.segmented_image[mask] = new_color

    # After updating colors, trigger an update to the segmented image display
    st.session_state.image_update_trigger += 1

def calculate_and_display_label_percentages() -> None:
    """Calculate and display label percentages"""
    final_labels = cv2.cvtColor(st.session_state.segmented_image, cv2.COLOR_BGR2GRAY)
    unique_labels, counts = np.unique(final_labels, return_counts=True)
    total_pixels = np.sum(counts)
    label_percentages = {int(label): (count / total_pixels) * 100 for label, count in zip(unique_labels, counts)}

    # Create a mapping from grayscale label to RGB color
    label_to_color = {}
    for label in unique_labels:
        mask = final_labels == label
        corresponding_color = st.session_state.segmented_image[mask][0]
        hex_color = f'#{corresponding_color[0]:02x}{corresponding_color[1]:02x}{corresponding_color[2]:02x}'
        label_to_color[int(label)] = hex_color

    st.write("Label Percentages:")
    for label, percentage in label_percentages.items():
        hex_color = label_to_color[label]
        color_box = f'<div style="display: inline-block; width: 20px; height: 20px; background-color: {hex_color}; margin-right: 10px;"></div>'
        st.markdown(f'{color_box} Label {label}: {percentage:.2f}%', unsafe_allow_html=True)

def main() -> None:
    st.title("PetroSeg")
    st.info("""
    - **Training Epochs**: Higher values will lead to fewer segments but may take more time.
    - **Image Size**: For better efficiency, upload small-sized images.
    - **Cache**: For best results, clear the cache between different image uploads. You can do this from the menu in the top-right corner.
    """)

    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    # Initialize session state if not already initialized
    if 'segmented_image' not in st.session_state:
        st.session_state.segmented_image = None
    if 'new_colors' not in st.session_state:
        st.session_state.new_colors = {}
    if 'image_update_trigger' not in st.session_state:
        st.session_state.image_update_trigger = 0

    # Define params before using it
    params = get_parameters_from_sidebar()

    uploaded_image = st.sidebar.file_uploader("Upload an image", type=["jpg", "png", "jpeg", "bmp", "tiff", "webp"])
    if uploaded_image:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        if image is None:
            st.error("Error loading image. Please check the file and try again.")
            return

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image_rgb, caption='Original Image', use_column_width=True)

        # Use the target size specified by the user
        target_size = params['target_size']
        image_resized = resize_image(image_rgb, target_size)

        if st.sidebar.button("Start Segmentation"):
            st.session_state.segmented_image = perform_custom_segmentation(image_resized, params)

        if st.sidebar.button("Change Colors"):
            randomize_colors()

        if st.session_state.segmented_image is not None:
            handle_color_picking()
            display_segmentation_results()
            calculate_and_display_label_percentages()
            download_image(st.session_state.segmented_image, 'segmented_image.png')

if __name__ == "__main__":
    main()
