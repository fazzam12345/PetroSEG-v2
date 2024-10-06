import cv2
import tempfile
from io import BytesIO
import streamlit as st
import numpy as np

def resize_image(image, target_size):
    """Resize image while maintaining aspect ratio."""
    original_height, original_width = image.shape[:2]
    target_width, target_height = target_size

    # Calculate the aspect ratio of the original image
    aspect_ratio = original_width / original_height

    # Determine the new dimensions while maintaining the aspect ratio
    if target_width / target_height > aspect_ratio:
        new_height = target_height
        new_width = int(new_height * aspect_ratio)
    else:
        new_width = target_width
        new_height = int(new_width / aspect_ratio)

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return resized_image

def automatically_change_segment_colors(segmented_image):
    # Generate a unique color for each segment
    unique_labels = np.unique(segmented_image.reshape(-1, 3), axis=0)
    new_colors = np.random.randint(0, 256, (len(unique_labels), 3), dtype=np.uint8)
    
    # Apply the new colors to the segmented image
    for i, label in enumerate(unique_labels):
        mask = np.all(segmented_image == label, axis=-1)
        segmented_image[mask] = new_colors[i]
    
    return segmented_image

def download_image(image_array, file_name):
    try:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        success = cv2.imwrite(temp_file.name, image_array)
        if not success:
            st.error("Could not save image.")
            return
        with open(temp_file.name, 'rb') as f:
            bytes = f.read()
        st.download_button(
            label="Download Image",
            data=BytesIO(bytes),
            file_name=file_name,
            mime='image/png',
        )
    except Exception as e:
        st.error(f"An error occurred: {e}")