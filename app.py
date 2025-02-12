import streamlit as st
import os
# Install libGL.so.1 if it's missing
if not os.path.isfile('/usr/lib/x86_64-linux-gnu/libGL.so.1'):
    os.system('sudo apt-get update && sudo apt-get install -y libgl1-mesa-glx')
    
from ultralytics import YOLO
from PIL import Image
import numpy as np
import gdown



# Function to download the model from Google Drive
def download_model():
    url = 'https://drive.google.com/uc?id=1ApmwcS9hCqxtDesn_B5B7AumqnzBRPEn'  # Replace FILE_ID with the actual ID from the shareable link
    output = 'best.pt'
    gdown.download(url, output, quiet=False)

# Download the model if it doesn't exist
if not os.path.exists('best.pt'):
    download_model()

# Load YOLO model
model = YOLO('best.pt')

# Streamlit app
st.title("YOLO Segmentation Demo")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    
    # Run model
    results = model(np.array(image))
    im_array = results[0].plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # Convert to RGB PIL image

    # Display results
    st.image(im, caption='Processed Image', use_column_width=True)

    # Display original image
    st.image(image, caption='Original Image', use_column_width=True)
