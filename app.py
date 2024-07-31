import streamlit as st
import os
# Install libGL.so.1 if it's missing
if not os.path.isfile('/usr/lib/x86_64-linux-gnu/libGL.so.1'):
    os.system('sudo apt-get update && sudo apt-get install -y libgl1-mesa-glx')
    
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load YOLO model
model_path = 'best.pt'
model = YOLO(model_path)

# Streamlit app
st.title("Baseboard YOLO-Seg Model")

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
