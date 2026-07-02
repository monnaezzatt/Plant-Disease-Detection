

import json

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
import gdown


st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

IMG_SIZE = (64, 64)
MODEL_PATH = "plant_disease_model.keras"
CLASS_NAMES_PATH = "class_names.json"



@st.cache_resource(show_spinner="Downloading & loading model...")
def load_model():

    if not os.path.exists(MODEL_PATH):
        gdown.download(
            "https://drive.google.com/uc?id=1ABfmnXOOBtJZ38I-jXS-RcVyY4Xsw4DA",
            MODEL_PATH,
            quiet=False
        )

    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_class_names():
    with open(CLASS_NAMES_PATH, "r") as f:
        return json.load(f)


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMG_SIZE)
    array = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def format_class_name(raw_name: str) -> str:
    return raw_name.replace("___", " - ").replace("_", " ")


with st.sidebar:
    st.title("🌿 About")
    st.write(
        "This app uses a **Multi-Layer Perceptron (MLP)** neural network "
        "trained on the [New Plant Diseases Dataset]"
        "(https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset) "
        "to classify plant leaf images into 38 healthy/diseased categories."
    )
    st.markdown("---")
    st.subheader("How to use")
    st.write(
        "1. Upload a clear photo of a single plant leaf.\n"
        "2. Wait for the model to process it.\n"
        "3. View the predicted disease and confidence score."
    )
    st.markdown("---")
    st.caption("Final Project — Machine Learning Course")


st.title("🌿 Plant Disease Classifier")
st.write(
    "Upload a leaf image and the model will predict whether the plant is "
    "healthy or identify the disease affecting it."
)

model = load_model()
class_names = load_class_names()

uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG",
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1.3])

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Prediction")
        with st.spinner("Analyzing image..."):
            input_array = preprocess_image(image)
            predictions = model.predict(input_array, verbose=0)[0]

        top_idx = int(np.argmax(predictions))
        top_class = format_class_name(class_names[top_idx])
        top_confidence = float(predictions[top_idx])

        if "healthy" in class_names[top_idx].lower():
            st.success(f"**Result:** {top_class}")
        else:
            st.error(f"**Result:** {top_class}")

        st.metric("Confidence", f"{top_confidence * 100:.2f}%")
        st.progress(top_confidence)

        st.markdown("#### Top 5 Predictions")
        top5_idx = np.argsort(predictions)[::-1][:5]
        for idx in top5_idx:
            name = format_class_name(class_names[idx])
            conf = float(predictions[idx])
            st.write(f"{name} — {conf * 100:.2f}%")
            st.progress(conf)
else:
    st.info("👆 Upload an image above to get a prediction.")

st.markdown("---")
st.caption(
    "Model: Multi-Layer Perceptron (MLP) trained on 64x64 RGB leaf images "
    "across 38 classes."
)
