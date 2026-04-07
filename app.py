import streamlit as st
from PIL import Image
import easyocr
import numpy as np
import cv2

st.title("✍️ Handwritten Text Recognition")

@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'], gpu=False)

reader = load_reader()

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image)

    img = np.array(image)

    # 🔥 RESIZE (MOST IMPORTANT FIX)
    img = cv2.resize(img, (600, 400))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    try:
        result = reader.readtext(gray, detail=0)  # lighter
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

    if result:
        text = " ".join(result)
        st.success(f"Prediction: {text}")
    else:
        st.error("❌ No text detected")