import streamlit as st
import os
import cv2
import numpy as np
from skimage import segmentation
import torch
import torch.nn as nn
from PIL import Image
import base64
from io import BytesIO
import tempfile

if torch.cuda.is_available():
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)


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



@st.cache_data
def perform_custom_segmentation(image, params):
    class Args(object):
        def __init__(self, params):
            self.train_epoch = params.get('train_epoch', 2 ** 3)
            self.mod_dim1 = params.get('mod_dim1', 64)
            self.mod_dim2 = params.get('mod_dim2', 32)
            self.gpu_id = params.get('gpu_id', 0)
            self.min_label_num = params.get('min_label_num', 6)
            self.max_label_num = params.get('max_label_num', 256)

    args = Args(params)

    class MyNet(nn.Module):
        def __init__(self, inp_dim, mod_dim1, mod_dim2):
            super(MyNet, self).__init__()
            self.seq = nn.Sequential(
                nn.Conv2d(inp_dim, mod_dim1, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(mod_dim1),
                nn.ReLU(inplace=True),
                nn.Conv2d(mod_dim1, mod_dim2, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(mod_dim2),
                nn.ReLU(inplace=True),
                nn.Conv2d(mod_dim2, mod_dim1, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(mod_dim1),
                nn.ReLU(inplace=True),
                nn.Conv2d(mod_dim1, mod_dim2, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(mod_dim2),
            )

        def forward(self, x):
            return self.seq(x)

    torch.cuda.manual_seed_all(1943)
    np.random.seed(1943)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    '''segmentation ML'''
    seg_map = segmentation.felzenszwalb(image, scale=15, sigma=0.06, min_size=14)
    seg_map = seg_map.flatten()
    seg_lab = [np.where(seg_map == u_label)[0]
               for u_label in np.unique(seg_map)]

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    tensor = image.transpose((2, 0, 1))
    tensor = tensor.astype(np.float32) / 255.0
    tensor = tensor[np.newaxis, :, :, :]
    tensor = torch.from_numpy(tensor).to(device)

    model = MyNet(inp_dim=3, mod_dim1=args.mod_dim1, mod_dim2=args.mod_dim2).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-2, momentum=0.9)

    image_flatten = image.reshape((-1, 3))
    color_avg = np.random.randint(255, size=(args.max_label_num, 3))
    show = image

    # Display segmenting progress
    progress_bar = st.progress(0)

    for batch_idx in range(args.train_epoch):
        optimizer.zero_grad()
        output = model(tensor)[0]
        output = output.permute(1, 2, 0).view(-1, args.mod_dim2)
        target = torch.argmax(output, 1)
        im_target = target.data.cpu().numpy()

        for inds in seg_lab:
            u_labels, hist = np.unique(im_target[inds], return_counts=True)
            im_target[inds] = u_labels[np.argmax(hist)]

        target = torch.from_numpy(im_target)
        target = target.to(device)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        un_label, lab_inverse = np.unique(im_target, return_inverse=True, )
        if un_label.shape[0] < args.max_label_num:
            img_flatten = image_flatten.copy()
            if len(color_avg) != un_label.shape[0]:
                color_avg = [np.mean(img_flatten[im_target == label], axis=0, dtype=int) for label in un_label]
            for lab_id, color in enumerate(color_avg):
                img_flatten[lab_inverse == lab_id] = color
            show = img_flatten.reshape(image.shape)

        # Update progress bar
        progress = (batch_idx + 1) / args.train_epoch
        progress_bar.progress(progress)

    return show


# Initialize session state
def initialize_session_state():
    st.session_state.setdefault('segmented_image', None)
    st.session_state.setdefault('new_colors', {})

# Get parameters from sidebar
def get_parameters_from_sidebar():
    st.sidebar.header("Segmentation Parameters")
    param_names = ['train_epoch', 'mod_dim1', 'mod_dim2', 'min_label_num', 'max_label_num']
    param_values = [(1, 100, 8), (1, 128, 64), (1, 128, 32), (1, 20, 6), (1, 500, 256)]
    params = {name: st.sidebar.slider(name.replace('_', ' ').title(), *values) for name, values in zip(param_names, param_values)}
    return params

# Display segmentation results
def display_segmentation_results():
    st.image(st.session_state.segmented_image, caption='Updated Segmented Image', use_column_width=True)   

# Handle color picking and other functionalities
def handle_color_picking():
    unique_labels = np.unique(st.session_state.segmented_image.reshape(-1, 3), axis=0)
    old_colors = np.array(list(st.session_state.new_colors.keys()))
    new_colors = np.array(list(st.session_state.new_colors.values()))

    for i, label in enumerate(unique_labels):
        hex_label = f'#{label[0]:02x}{label[1]:02x}{label[2]:02x}'
        new_color = st.color_picker(f"Choose a new color for label {i}", value=hex_label, key=f"label_{i}")

        new_color_rgb = tuple(int(new_color.lstrip('#')[j:j+2], 16) for j in (0, 2, 4))
        st.session_state.new_colors[tuple(label)] = new_color_rgb

    for old_color, new_color in st.session_state.new_colors.items():
        mask = np.all(st.session_state.segmented_image == np.array(old_color), axis=-1)
        st.session_state.segmented_image[mask] = new_color



def calculate_and_display_label_percentages():
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


# Main function
def main():
    # Initialize session state
    initialize_session_state()

    # Streamlit UI
    st.title("Unsupervised Segmentation App")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png"])
# Main function
def main():
    # Initialize session state
    initialize_session_state()

    # Streamlit UI
    st.title("Unsupervised Segmentation App")
    st.info("""
    - **Training Epochs**: Higher values will lead to fewer segments but may take more time.
    - **Image Size**: For better efficiency, upload small-sized images.
    """)
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png"])

    if uploaded_image:
        im = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), 1)
        params = get_parameters_from_sidebar()

        if st.sidebar.button("Start Segmentation"):
            st.session_state.segmented_image = perform_custom_segmentation(im, params)

        if st.session_state.segmented_image is not None:
            handle_color_picking()
            display_segmentation_results()
            calculate_and_display_label_percentages()
            download_image(st.session_state.segmented_image, 'segmented_image.png') 

if __name__ == "__main__":
    main()
