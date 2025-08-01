import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import os

# Constants
IMG_SIZE = (224, 224)
MODEL_PATH = "tree_species_model.h5"

# Load Model
@st.cache_resource
def load_efficientnet_model():
    model = load_model(MODEL_PATH)
    return model

# Predict function
def predict_species(image, model, class_labels):
    img = image.resize(IMG_SIZE)
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)[0]
    top_3_indices = predictions.argsort()[-3:][::-1]
    top_3 = [(class_labels[i], predictions[i]) for i in top_3_indices]
    return top_3

# Main App
def main():
    st.title("🌳 Tree Species Classifier")
    model = load_efficientnet_model()

    st.write("Upload a tree image to classify its species.")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    # Dummy labels (Update these from your dataset)
    class_labels = sorted([
        "Acacia", "Aloe", "Ashoka", "Bamboo", "Banyan", "Bael", "Bottlebrush",
        "Coconut", "Drumstick", "Eucalyptus", "Ficus", "Flame", "Gulmohar",
        "Guava", "IndianTulip", "Jamun", "Jasmine", "Lemon", "Mahogany",
        "Mango", "Neem", "Peepal", "Pine", "Pomegranate", "RainTree",
        "Rosewood", "Seesam", "Siris", "Tamarind", "Teak"
    ])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        st.subheader("🔍 Classification Result")
        with st.spinner("Predicting..."):
            top_3 = predict_species(image, model, class_labels)
            top_1 = top_3[0]
            st.success(f"**Top Prediction:** {top_1[0]} ({top_1[1]*100:.2f}% confidence)")
            st.write("Top 3 Predictions:")
            for label, prob in top_3:
                st.write(f"- {label}: {prob:.2%}")

if __name__ == "__main__":
    main()
