import cv2
import streamlit as st
import numpy as np
from PIL import Image


def threshold(image, hul, huh):
    return cv2.threshold(
        image,
        hul,
        huh,
        cv2.THRESH_BINARY_INV
        )


def main_loop():
    st.title("OpenCV Demo App")
    st.subheader("This app allows you to find threshold to convert color Image to binary Image!")
    st.text("We use OpenCV and Streamlit for this demo")

    hul = st.sidebar.slider("Max", min_value=0, max_value=255)
    huh = st.sidebar.slider("Min", min_value=0, max_value=255, value=0)
##    apply_enhancement_filter = st.sidebar.checkbox('Enhance Details')

    image_file = st.file_uploader(
        "Upload Your Image",
        type=['jpg', 'png', 'jpeg']
        )
    if not image_file:
        return None

    original_image = Image.open(image_file)
    original_image = np.array(original_image)

    ret, processed_image = threshold(original_image, hul, huh)
##
##    if apply_enhancement_filter:
##        processed_image = enhance_details(processed_image)

    st.text("Original Image vs Processed Image")
    st.image([original_image, processed_image])


if __name__ == '__main__':
    main_loop()
##streamlit run proj.py
