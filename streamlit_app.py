import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import EfficientNetB0
import os

# Constants
IMG_HEIGHT = 224
IMG_WIDTH = 224
NUM_CLASSES = 30  # Change this if your model was trained on fewer/more classes

# Dummy class labels (replace with actual ones if available)
class_labels = [f"Class {i+1}" for i in range(NUM_CLASSES)]

# Load the model with EfficientNetB0 as a custom object
@st.cache_resource
def load_model_fully():
    model = load_model("tree_species_model.h5", custom_objects={"EfficientNetB0": EfficientNetB0})
    return model

model = load_model_fully()

# Preprocessing function
def preprocess_image(img, target_size=(224, 224)):
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Streamlit UI
st.title("🌿 Tree Species Classifier")
st.write("Upload a tree leaf image to predict the species using a CNN model.")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    st.write("⏳ Classifying...")

    preprocessed = preprocess_image(img)
    prediction = model.predict(preprocessed)[0]

    pred_index = np.argmax(prediction)
    pred_class = class_labels[pred_index]
    confidence = prediction[pred_index] * 100

    st.success(f"✅ Predicted Class: **{pred_class}**")
    st.write(f"🔍 Confidence: **{confidence:.2f}%**")

    st.subheader("🔝 Top 3 Predictions:")
    top_3 = prediction.argsort()[-3:][::-1]
    for i in top_3:
        st.write(f"{class_labels[i]} — {prediction[i]:.2%}")
