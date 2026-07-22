import streamlit as st
from pathlib import Path
from src.predict import predict_image


st.set_page_config(page_title="CatDog Classifier", page_icon="🐶")
st.title("CatDog Classifier")
st.write("Upload an image to classify it as a cat or a dog.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_path = Path("/tmp") / uploaded_file.name
    image_path.write_bytes(uploaded_file.getvalue())
    st.image(image_path, caption="Uploaded image", use_container_width=True)

    label, confidence = predict_image(str(image_path))
    st.success(f"Prediction: {label} ({confidence:.2%} confidence)")
