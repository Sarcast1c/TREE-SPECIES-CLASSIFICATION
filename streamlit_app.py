import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input

# Constants
IMG_SIZE = (224, 224)
MODEL_PATH = "tree_species_model.h5"

# Load model from architecture and weights
@st.cache_resource
def load_efficientnet_model():
    base_model = EfficientNetB0(include_top=False, input_shape=(224, 224, 3), weights='imagenet')
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(30, activation='softmax')  # Make sure this matches your num_classes
    ])
    model.load_weights(MODEL_PATH)
    return model

def predict_species(image, model, class_labels):
    img = image.resize(IMG_SIZE)
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)[0]
    top_3_indices = predictions.argsort()[-3:][::-1]
    top_3 = [(class_labels[i], predictions[i]) for i in top_3_indices]
    return top_3

def main():
    st.title("🌳 Tree Species Classifier")
    model = load_efficientnet_model()

    st.write("Upload a tree image to classify its species.")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

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
