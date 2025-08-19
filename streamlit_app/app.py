import streamlit as st
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the updated .keras model
model = load_model('models/cnn_from_scratch.keras')

# Automatically get class labels from folder names
class_labels = sorted(os.listdir('data/train'))

st.title("🐟 Fish Image Classifier")

uploaded = st.file_uploader("Upload a fish image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    img = image.load_img(uploaded, target_size=(224, 224))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_label = class_labels[predicted_index]
    confidence = round(100 * np.max(predictions), 2)

    st.success(f"Prediction: **{predicted_label}** ({confidence}% confidence)")