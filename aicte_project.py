import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import EfficientNetB0
from PIL import Image
import os

# -------------------------------
# Load Pretrained Model
# -------------------------------
@st.cache_resource
def load_trained_model():
    model = load_model("tree_species_model.h5", custom_objects={'EfficientNetB0': EfficientNetB0})
    return model

model = load_trained_model()

# -------------------------------
# Image Preprocessing Function
# -------------------------------
def preprocess_image(img, target_size=(224, 224)):
    if isinstance(img, str):
        img = image.load_img(img, target_size=target_size)
    else:
        img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# -------------------------------
# Class Labels (Replace with yours)
# -------------------------------
class_labels = os.listdir("TREE-SPECIES-CLASSIFICATION/Tree_Species_Dataset") if os.path.exists("TREE-SPECIES-CLASSIFICATION/Tree_Species_Dataset") else [f"Class {i}" for i in range(30)]

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🌳 Tree Species Classification")

st.write("Upload a tree leaf image, and the model will predict the tree species.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    st.write("⏳ Classifying...")
    preprocessed = preprocess_image(img)
    prediction = model.predict(preprocessed)
    predicted_class = class_labels[np.argmax(prediction)]

    st.success(f"✅ Predicted Species: **{predicted_class}**")
