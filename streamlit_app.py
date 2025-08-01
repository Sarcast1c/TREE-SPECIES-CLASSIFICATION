import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# --- Configuration ---
# Assuming your model file is named 'improved_cnn_model.h5' or 'basic_cnn_tree_species.h5'
# and is in the same directory as your Streamlit app, or provide the correct path.
MODEL_PATH = 'improved_cnn_model.h5' # Change this if you used the basic CNN model

# Assuming your dataset structure allows extracting class names from directory names.
# If not, you might need to hardcode the class names list here in the correct order
# as per your model's output.
# Example: CLASS_NAMES = ['amla', 'asopalav', ...]

# You might need to adjust the path to your dataset depending on where you run the Streamlit app.
# If the dataset is not needed for prediction (only the model and class names),
# you can remove the dataset loading part and hardcode CLASS_NAMES.
REPO_PATH = os.path.join("TREE-SPECIES-CLASSIFICATION", "Tree_Species_Dataset")


# --- Load Model and Class Names ---
@st.cache_resource # Cache the model loading for better performance
def load_my_model(model_path):
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource # Cache class names loading
def get_class_names(repo_path):
    try:
        # This assumes the directory names in REPO_PATH are your class names
        class_names = sorted(os.listdir(repo_path))
        return class_names
    except Exception as e:
        st.error(f"Error getting class names from {repo_path}: {e}")
        # Fallback if directory listing fails - hardcode class names if known
        # return ['amla', 'asopalav', ...] # Example hardcoded list
        return None


model = load_my_model(MODEL_PATH)
class_names = get_class_names(REPO_PATH)

if model is None or class_names is None:
    st.stop() # Stop the app if model or class names couldn't be loaded


# --- Prediction Function ---
def predict_image(img, model, class_names, img_height=224, img_width=224):
    # Preprocess the image
    img = img.resize((img_width, img_height)) # Resize the image
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Rescale the image

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class_name = class_names[predicted_class_index]
    confidence = predictions[0][predicted_class_index]

    return predicted_class_name, confidence

# --- Streamlit App Interface ---
st.title("Tree Species Classification")

st.write("Upload an image of a tree leaf to predict its species.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = image.load_img(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Make prediction
    if model is not None and class_names is not None:
        with st.spinner("Predicting..."):
            predicted_species, confidence = predict_image(img, model, class_names)

        st.write(f"**Predicted Species:** {predicted_species}")
        st.write(f"**Confidence:** {confidence:.2f}")
    else:
        st.error("Model or class names not loaded. Cannot make prediction.")

# --- Instructions for Running ---
st.sidebar.header("How to run this app")
st.sidebar.markdown("""
1. Save the code above as a Python file (e.g., `app.py`).
2. Make sure your trained model file (`improved_cnn_model.h5` or `basic_cnn_tree_species.h5`) is in the same directory as `app.py`.
3. Open your terminal or command prompt.
4. Navigate to the directory where you saved the files.
5. Run the command: `streamlit run app.py`
6. Your web browser should open with the Streamlit app.
""")
