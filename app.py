import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import pickle
from PIL import Image

# Load models
@st.cache_resource
def load_models():
    unet = tf.keras.models.load_model("unet_model.h5")
    clf = tf.keras.models.load_model("alzheimer_classifier.h5")
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return unet, clf, scaler

unet_model, clf_model, scaler = load_models()

# Constants
IMG_SIZE = 128

st.title("🧠 Alzheimer’s Disease Prediction Interface")
st.write("Upload an MRI and enter clinical data to predict disease progression.")

# Upload Image
uploaded_file = st.file_uploader("Upload Brain MRI Image", type=["jpg", "png", "jpeg"])

# Input clinical data
age = st.number_input("Age", min_value=40, max_value=100, value=70)
gender = st.selectbox("Gender", ["Male", "Female"])
mmse = st.slider("MMSE Score", 0, 30, 25)
cdr = st.slider("CDR Score", 0.0, 3.0, 0.5, step=0.1)
edu_years = st.slider("Years of Education", 0, 25, 12)

if st.button("Predict"):
    if uploaded_file:
        # Process image
        image = Image.open(uploaded_file).convert('L')
        image = image.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(image) / 255.0
        img_input = np.expand_dims(img_array, axis=(0, -1))

        # Segment image using U-Net
        segmented = unet_model.predict(img_input)
        flat_segmented = segmented.reshape(1, -1)

        # Process clinical features
        gender_enc = 1 if gender == "Male" else 0
        clinical_input = np.array([[age, gender_enc, mmse, cdr, edu_years]])
        clinical_scaled = scaler.transform(clinical_input)

        # Concatenate
        final_input = np.concatenate([flat_segmented, clinical_scaled], axis=1)

        # Predict
        prediction = clf_model.predict(final_input)[0][0]
        st.success(f"🧪 Prediction Score: {prediction:.4f}")
        if prediction >= 0.5:
            st.error("🧠 Likely Alzheimer's Disease Detected.")
        else:
            st.success("✅ Normal Brain Condition.")
    else:
        st.warning("Please upload an MRI image first.")
