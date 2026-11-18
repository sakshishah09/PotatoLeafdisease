
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model(
    "potato_cnn_model.keras"
)

# Load class names
with open("classification_report.txt", "r") as file:
    class_names = file.read().splitlines()

st.title("🌿 Potato Disease Detection")

st.write("Upload a potato leaf image to detect disease.")

# Upload image
uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Leaf")

    if st.button("Predict Disease"):

        # Resize image
        image = image.resize((128, 128))

        # Convert image to array
        img_array = np.array(image, dtype=np.float32)

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        # Model prediction
        predictions = model.predict(img_array, verbose=0)[0]

        # Find highest probability
        best_index = np.argmax(predictions)

        predicted_class = class_names[best_index]
        confidence = predictions[best_index]

        st.success(f"Prediction: {predicted_class}")

        st.write(
            f"Confidence: {confidence * 100:.2f}%"
        )

        st.subheader("All Class Probabilities")

        for name, score in zip(class_names, predictions):

            st.write(
                f"{name}: {score * 100:.2f}%"
            )

            st.progress(float(score))
