from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐶", layout="centered")
st.title("Cat vs Dog Image Classifier")
st.write("Upload an image to test the trained classifier.")

uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded image", use_container_width=True)
    st.info("Model inference will be wired in once the trained model is available.")
else:
    st.info("Please upload an image to begin.")
