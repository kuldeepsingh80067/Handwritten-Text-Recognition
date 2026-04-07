import streamlit as st
from PIL import Image
import numpy as np
import cv2
import easyocr
import re

# -----------------------------
# PAGE SETUP
# -----------------------------
st.set_page_config(page_title="Smart OCR", page_icon="✍️")
st.title("✍️ Smart Text Recognition (Printed + Handwritten)")

# -----------------------------
# LOAD OCR
# -----------------------------
@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'], gpu=False)

reader = load_reader()

# -----------------------------
# INPUT
# -----------------------------
option = st.radio("Choose Input:", ["Upload Image", "Use Camera"])

image = None

if option == "Upload Image":
    file = st.file_uploader("Upload image", type=["png","jpg","jpeg"])
    if file:
        image = Image.open(file)

else:
    cam = st.camera_input("Take photo")
    if cam:
        image = Image.open(cam)

# -----------------------------
# SMART OCR FUNCTION
# -----------------------------
def extract_text_smart(img):

    results = []

    # 1️⃣ ORIGINAL IMAGE (BEST FOR PRINTED)
    r1 = reader.readtext(img, detail=1, paragraph=True)
    t1 = " ".join([r[1] for r in r1])
    s1 = sum([r[2] for r in r1]) if r1 else 0
    results.append((t1, s1))

    # 2️⃣ GRAYSCALE
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    r2 = reader.readtext(gray, detail=1, paragraph=True)
    t2 = " ".join([r[1] for r in r2])
    s2 = sum([r[2] for r in r2]) if r2 else 0
    results.append((t2, s2))

    # 3️⃣ LIGHT PREPROCESS (FOR HANDWRITING)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    _, thresh = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    r3 = reader.readtext(thresh, detail=1, paragraph=True)
    t3 = " ".join([r[1] for r in r3])
    s3 = sum([r[2] for r in r3]) if r3 else 0
    results.append((t3, s3))

    # BEST RESULT
    best_text, best_score = max(results, key=lambda x: x[1])

    return best_text, best_score


# -----------------------------
# PROCESS
# -----------------------------
if image:

    st.image(image, caption="Input Image")

    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    with st.spinner("Extracting text..."):
        text, score = extract_text_smart(img)

    # CLEAN TEXT
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)

    # -----------------------------
    # OUTPUT
    # -----------------------------
    st.subheader("📊 Result")
    st.metric("Confidence", round(score, 2))

    st.success("Extracted Text")
    st.write(text if text else "No text detected ❌")

    st.code(text)

    st.download_button(
        "📥 Download Text",
        text,
        "output.txt"
    )
