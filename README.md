# Unsupervised Segmentation App with Streamlit and PyTorch

## Table of Contents
1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [How to Run](#how-to-run)
5. [Code Explanation](#code-explanation)
6. [Contributing](#contributing)
7. [License](#license)

---

## Introduction 🌟
This project is a web application built using Streamlit and PyTorch. It performs unsupervised segmentation on uploaded images. The segmented image can be downloaded, and the colors of the segments can be customized.

---

## Requirements 📋
- Python 3.x
- Streamlit
- PyTorch
- OpenCV
- NumPy
- scikit-image
- PIL
- base64

---

## Installation 🛠️

1. **Clone the repository**
    ```bash
    git clone https://github.com/your-repo/unsupervised-segmentation.git
    ```
2. **Navigate to the project directory**
    ```bash
    cd unsupervised-segmentation
    ```
3. **Install the required packages**
    ```bash
    pip install -r requirements.txt
    ```

---

## How to Run 🚀

1. **Navigate to the project directory**
    ```bash
    cd unsupervised-segmentation
    ```
2. **Run the Streamlit app**
    ```bash
    streamlit run app.py
    ```

---

## Code Explanation 📝

### Importing Libraries
- **Streamlit**: For creating the web application.
- **PyTorch**: For the neural network model.
- **OpenCV**: For image processing.
- **scikit-image**: For initial segmentation.
- **NumPy**: For numerical operations.
- **PIL**: For image handling.
- **base64**: For encoding the image for download.

### Helper Functions
- `get_image_download_link`: Generates a download link for the segmented image.
- `perform_custom_segmentation`: Performs the segmentation using a neural network.

### Streamlit UI
- File uploader and sliders for segmentation parameters are created using Streamlit.

### Main Function
- Orchestrates the entire flow of the application, from UI creation to segmentation and customization.

---

## Contributing 🤝
Feel free to open issues and pull requests!

---

## License 📜
This project is licensed under the MIT License.

