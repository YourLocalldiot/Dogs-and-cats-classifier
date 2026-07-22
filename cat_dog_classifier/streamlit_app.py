from pathlib import Path
import time

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

from src.prediction import load_class_names, load_model_artifact, predict_image, read_image

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "final_model.keras"
CLASS_NAMES_PATH = ROOT / "models" / "class_names.npy"


@st.cache_resource
def load_trained_model() -> object:
    """Load the trained TensorFlow model once and reuse it for all inferences."""
    return load_model_artifact(MODEL_PATH)


@st.cache_data
def load_labels() -> np.ndarray:
    """Load class names once and reuse them across the app."""
    return load_class_names(CLASS_NAMES_PATH)


def save_temp_image(image: Image.Image, name: str) -> Path:
    """Persist a temporary image file for prediction helpers that expect a file path."""
    temp_dir = ROOT / "tmp"
    temp_dir.mkdir(exist_ok=True)
    temp_path = temp_dir / name
    image.save(temp_path)
    return temp_path


def render_prediction_result(image_path: Path, label: str, confidence_pct: float, probability: float) -> None:
    """Display the prediction output with styling and progress bar."""
    st.image(Image.open(image_path), caption="Preview", use_container_width=True)

    st.subheader("Prediction Result")
    if label.lower() == "cat":
        st.success(f"🧡 Predicted class: {label}")
    elif label.lower() == "dog":
        st.success(f"🐶 Predicted class: {label}")
    else:
        st.warning(f"Predicted class: {label}")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.metric("Confidence", f"{confidence_pct:.2f}%")
        st.progress(min(confidence_pct / 100.0, 1.0))
    with col2:
        st.metric("Probability", f"{probability:.4f}")

    st.caption("Inference completed successfully.")


def handle_image_input(image: Image.Image, source_name: str) -> None:
    """Run inference for a single image and display the result."""
    try:
        image_path = save_temp_image(image, f"{source_name}.png")
        start = time.perf_counter()
        label, confidence_pct, probability = predict_image(image_path, MODEL_PATH, CLASS_NAMES_PATH)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        render_prediction_result(image_path, label, confidence_pct, probability)
        st.caption(f"Inference time: {elapsed_ms:.2f} ms")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        st.error(f"Unable to process the image: {exc}")


st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐶", layout="wide")
st.title("🐶 Cat vs Dog Classifier")
st.markdown("Upload an image or use your camera to classify it as a cat or dog with a TensorFlow model.")

with st.sidebar:
    st.header("About")
    st.write("This app uses a trained TensorFlow classifier to identify whether an image contains a cat or a dog.")

    st.header("Model")
    st.write(f"- Model path: {MODEL_PATH.name}")
    st.write("- Architecture: EfficientNetB0 with transfer learning")
    st.write("- Output classes: Cat / Dog")

    st.header("Instructions")
    st.write("1. Choose a tab below.")
    st.write("2. Upload an image or capture one with your camera.")
    st.write("3. Review the prediction and confidence score.")

# Load the model and labels once on startup.
model = load_trained_model()
labels = load_labels()

if model is not None and labels is not None:
    st.success("Model ready for inference.")
else:
    st.error("The trained model could not be loaded.")

tab_upload, tab_camera = st.tabs(["📤 Upload Image", "📷 Camera Capture"])

with tab_upload:
    st.header("Upload an image")
    uploaded_file = st.file_uploader("Choose a JPG, JPEG, or PNG image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            handle_image_input(image, "uploaded")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            st.error(f"The selected file could not be read: {exc}")

with tab_camera:
    st.header("Capture an image")
    captured_image = st.camera_input("Take a photo")
    if captured_image is not None:
        try:
            image = Image.open(captured_image).convert("RGB")
            handle_image_input(image, "camera")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            st.error(f"The captured image could not be processed: {exc}")
