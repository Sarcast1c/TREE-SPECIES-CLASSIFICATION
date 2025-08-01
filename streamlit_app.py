import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import img_to_array

IMG_HEIGHT = 224
IMG_WIDTH = 224
NUM_CLASSES = 30  # Update if you know your actual number of tree classes

# Dummy class labels
class_labels = [f"Class {i+1}" for i in range(NUM_CLASSES)]

@st.cache_resource
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.load_weights("tree_species_model.h5")
    return model

model = build_model()

def preprocess(img):
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

st.title("🌳 Tree Species Classifier")
st.write("Upload a tree image to predict its species.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)

        input_array = preprocess(img)
        predictions = model.predict(input_array)[0]

        top_3 = predictions.argsort()[-3:][::-1]
        st.subheader("🔝 Top 3 Predictions:")
        for idx in top_3:
            st.write(f"{class_labels[idx]} — {predictions[idx]:.2%}")

        st.success(f"🌲 Most Likely Species: **{class_labels[top_3[0]]}**")
    except Exception as e:
        st.error("❌ Error processing image.")
        st.exception(e)
