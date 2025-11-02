import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="Retinal Disease Classifier", page_icon="🩺")

st.title("🩺 Retinal Disease Classification")
st.write("Upload a retinal image and the model will predict the disease type using a trained MobileNetV2 model.")

# Load TensorFlow Lite model
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="MobileNetV2_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

# Get model input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# File uploader
uploaded_file = st.file_uploader("Choose a retinal image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼 Uploaded Retinal Image", use_container_width=True)

    # Preprocess the image
    image = image.resize((224, 224))
    input_data = np.expand_dims(np.array(image, dtype=np.float32) / 255.0, axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]

    # Define class names (update if you used different labels)
    class_names = [
        "Diabetic Retinopathy",
        "Glaucoma",
        "Cataract",
        "Age-related Macular Degeneration",
        "Normal"
    ]

    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    st.success(f"✅ **Prediction:** {predicted_class}")
    st.info(f"🎯 **Confidence:** {confidence:.2f}%")

    st.divider()
    st.caption("Model: MobileNetV2 | Framework: TensorFlow Lite | Developed by Uroosha Usman 💻")
