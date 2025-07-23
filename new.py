import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

# ✅ Must be the first Streamlit call
st.set_page_config(page_title="Alzheimer MRI Predictor", layout="centered")

# === Configuration ===
IMG_SIZE = 128
CLASS_NAMES = [
    "No Impairment",
    "Very Mild Impairment",
    "Mild Impairment",
    "Moderate Impairment"
]

# === Load Trained Model ===
@st.cache_resource
def load_cnn_model():
    model = load_model("alzheimer_image_classifier.h5")
    return model

model = load_cnn_model()

# === App UI ===
st.title("🧠 Alzheimer’s Disease Prediction from MRI")
st.markdown("Upload a grayscale brain MRI image to classify the stage of Alzheimer’s disease.")

# === File Uploader ===
uploaded_file = st.file_uploader("Upload MRI Image (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    # Resize and normalize
    resized_img = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    normalized_img = resized_img / 255.0
    input_img = np.expand_dims(normalized_img, axis=(0, -1))  # Shape: (1, 128, 128, 1)

    # Show uploaded image
    st.image(resized_img, caption="Uploaded MRI Image", use_column_width=False, width=300)

    # Predict
    if st.button("🔍 Predict"):
        prediction = model.predict(input_img)[0]
        predicted_class = CLASS_NAMES[np.argmax(prediction)]
        confidence = prediction[np.argmax(prediction)] * 100

        # Output
        st.success(f"**Prediction:** {predicted_class}")
        st.info("**Confidence Scores:**")
        for i, prob in enumerate(prediction):
            st.write(f"- {CLASS_NAMES[i]}: **{prob * 100:.2f}%**")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit and TensorFlow")
