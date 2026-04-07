import streamlit as st
from PIL import Image
import numpy as np
import cv2
import easyocr
import re

st.set_page_config(page_title="Advanced Handwritten OCR", layout="centered")
st.title("🚀 Advanced Handwritten Text Recognition")

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

# Initialize OCR reader once (faster)
@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'], gpu=False)

reader = load_reader()

# -----------------------------
# 🔥 MULTI PREPROCESSING
# -----------------------------
def preprocess_variants(img):
    variants = []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Variant 1: Normal threshold
    _, t1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(t1)

    # Variant 2: Adaptive threshold
    t2 = cv2.adaptiveThreshold(gray, 255,
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 2)
    variants.append(t2)

    # Variant 3: Blur + threshold
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, t3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(t3)

    return variants


# -----------------------------
# 🤖 OCR WITH CONFIDENCE
# -----------------------------
def run_ocr_best(variants):
    best_text = ""
    best_score = 0
    best_boxes = None

    for img in variants:
        result = reader.readtext(img)

        text = " ".join([r[1] for r in result])
        score = sum([r[2] for r in result]) if result else 0

        if score > best_score:
            best_score = score
            best_text = text
            best_boxes = result

    return best_text, best_score, best_boxes


if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Preprocessing
    variants = preprocess_variants(img)

    st.subheader("🧠 Trying Multiple Enhancements...")
    st.image(variants, caption=["Variant 1", "Variant 2", "Variant 3"], width=200)

    # OCR
    with st.spinner("Analyzing..."):
        text, score, boxes = run_ocr_best(variants)

    # Clean text
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)

    # -----------------------------
    # 📊 OUTPUT
    # -----------------------------
    st.success("✅ Best Prediction:")
    st.write(text if text else "No text detected ❌")

    st.info(f"Confidence Score: {round(score, 2)}")

    # -----------------------------
    # 📦 DRAW BOUNDING BOXES
    # -----------------------------
    if boxes:
        drawn = img.copy()

        for (bbox, txt, conf) in boxes:
            pts = np.array(bbox).astype(int)
            cv2.polylines(drawn, [pts], True, (0,255,0), 2)

        drawn = cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB)
        st.image(drawn, caption="Detected Text Regions")

    # -----------------------------
    # 💾 DOWNLOAD
    # -----------------------------
    st.download_button(
        "📥 Download Text",
        data=text,
        file_name="output.txt",
        mime="text/plain"
    )
