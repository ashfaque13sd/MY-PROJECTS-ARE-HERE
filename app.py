import streamlit as st
from tensorflow import keras
import numpy as np
import cv2

# Load pre-trained model
model = keras.models.load_model(
    r"C:/Users/LENOVO/Documents/my data/all projects/null class project/face expression recognition project/face_recognition_trained_model.keras"
)

emotions = ["surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"]

st.header("Face Expression Recognition")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is not None:
        image_resized = cv2.resize(image, (100, 100))  # adjust if needed
        image_normalized = image_resized / 255.0
        image_input = np.expand_dims(image_normalized, axis=0)

        try:
            prediction = model.predict(image_input)
            output = np.argmax(prediction[0])
            outcome = emotions[output]
            st.success(f"Emotion in the image is: {outcome}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image_rgb, caption="Uploaded Image")
    else:
        st.error("Could not read the image.")

