import streamlit as st
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import numpy as np
import cv2

st.set_page_config(page_title="AI OCR", page_icon="🧠")

st.title("🧠 Handwritten OCR (AI Powered)")

@st.cache_resource
def load_model():
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    return processor, model

processor, model = load_model()

def extract_text(image):
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    with torch.no_grad():
        ids = model.generate(pixel_values)
    text = processor.batch_decode(ids, skip_special_tokens=True)[0]
    return text

uploaded = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded:
    image = Image.open(uploaded)
    st.image(image)

    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2)

    pil_img = Image.fromarray(gray)

    with st.spinner("Reading..."):
        text = extract_text(pil_img)

    st.success("Done")
    st.code(text)
