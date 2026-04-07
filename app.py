import streamlit as st
from PIL import Image
import numpy as np
import cv2
import easyocr
import re

st.set_page_config(page_title="Smart OCR", page_icon="🧠")

st.markdown("""
    <h1 style='text-align: center;'>🧠 Smart OCR (Stable Version)</h1>
    <h4 style='text-align: center; color: gray;'>Developed by Kuldeep Singh</h4>
""", unsafe_allow_html=True)

st.markdown("Fast • Reliable • Works on Streamlit 🚀")

@st.cache_resource
def load_model():
    return easyocr.Reader(['en'], gpu=False)

reader = load_model()

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2)

    blur = cv2.GaussianBlur(gray, (3,3), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    return thresh

def extract_text(img):
    processed = preprocess(img)

    result = reader.readtext(processed, detail=1, paragraph=True)

    texts = []
    for r in result:
        try:
            texts.append(r[1])
        except:
            continue

    text = " ".join(texts)
    text = re.sub(r'\s+', ' ', text)

    return text

# ✅ NEW: USER CHOICE
option = st.radio("Choose input method:", ["Upload Image", "Use Camera"])

image = None

if option == "Upload Image":
    uploaded = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    if uploaded:
        image = Image.open(uploaded)

elif option == "Use Camera":
    camera = st.camera_input("📷 Take Photo")
    if camera:
        image = Image.open(camera)

# ------------------------

if image is not None:

    st.image(image, use_container_width=True)

    img = np.array(image)

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    with st.spinner("Reading text..."):
        text = extract_text(img)

    st.success("Done!")

    st.subheader("📄 Extracted Text")
    st.code(text if text else "⚠ No text detected")

    st.download_button("📥 Download", text, file_name="output.txt")

else:
    st.info("Upload or capture image")
