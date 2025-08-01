import streamlit as st
import numpy as np
from PIL import Image
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam

# ----------------------------
# Constants
# ----------------------------
IMG_HEIGHT = 224
IMG_WIDTH = 224
NUM_CLASSES = 30  # Replace with your actual number of tree species

# ----------------------------
# Class labels
# ----------------------------
# Optional: Replace with real class names if you have them
class_labels = [f"Class {i+1}" for i in range(NUM_CLASSES)]

# ----------------------------
# Load the trained model
# ----------------------------
@st.cache_resource
def load_model_weights():
    base_model = EfficientNetB0(include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), weights='imagenet')
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.load_weights("tree_species_model.h5")
    return model

model = load_model_weights()

# ----------------------------
# Image preprocessing
# ----------------------------
def preprocess_image(img, target_size=(224, 224)):
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize
    return img_array

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🌳 Tree Species Classification")
st.write("Upload an image of a tree leaf to classify the species using EfficientNetB0.")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.write("⏳ Classifying...")

    processed = preprocess_image(img)
    prediction = model.predict(processed)[0]
    predicted_index = np.argmax(prediction)
    predicted_class = class_labels[predicted_index]
    confidence = prediction[predicted_index] * 100

    st.success(f"✅ Predicted Species: **{predicted_class}**")
    st.info(f"📊 Confidence: {confidence:.2f}%")
