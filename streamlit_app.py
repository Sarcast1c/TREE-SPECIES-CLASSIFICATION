import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Constants
IMG_HEIGHT = 224
IMG_WIDTH = 224
NUM_CLASSES = 30  # Adjust this based on your dataset

# Dummy labels (update if you have real class names)
class_labels = [f"Class {i+1}" for i in range(NUM_CLASSES)]

@st.cache_resource
def load_model_safely():
    return load_model("tree_species_model.h5", compile=False)

model = load_model_safely()

def preprocess(img):
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

st.title("🌳 Tree Species Identifier")
st.write("Upload a tree image to predict the species.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)

        img_array = preprocess(img)
        predictions = model.predict(img_array)[0]

        pred_idx = np.argmax(predictions)
        pred_class = class_labels[pred_idx]
        confidence = predictions[pred_idx]

        st.success(f"✅ Predicted Species: **{pred_class}**")
        st.write(f"🔍 Confidence: **{confidence:.2%}**")

        st.subheader("🔝 Top 3 Predictions:")
        top_3 = predictions.argsort()[-3:][::-1]
        for i in top_3:
            st.write(f"{class_labels[i]} — {predictions[i]:.2%}")

    except Exception as e:
        st.error("⚠️ Error during prediction. Check if the uploaded image matches the expected input format.")
        st.exception(e)
