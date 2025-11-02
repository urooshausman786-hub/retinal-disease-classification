import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title="Retinal Disease Classification",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------------------- CUSTOM CSS ----------------------
st.markdown("""
    <style>
    body {
        background-color: #f5faff;
        font-family: 'Segoe UI', sans-serif;
    }
    .main-title {
        text-align: center;
        color: #004e92;
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 0px;
    }
    .sub-title {
        text-align: center;
        color: #444444;
        font-size: 18px;
        margin-top: -5px;
        margin-bottom: 30px;
        font-weight: 500;
    }
    .stButton>button {
        background-color: #004e92;
        color: white;
        border: None;
        border-radius: 10px;
        padding: 10px 24px;
        font-size: 16px;
        font-weight: 600;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #0078d7;
        color: white;
        transform: scale(1.05);
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------- TITLES ----------------------
st.markdown('<h1 class="main-title">🩺 Retinal Disease Classification System</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Upload a retinal image to detect possible eye diseases using Deep Learning (CNN - MobileNetV2 Model)</p>',
    unsafe_allow_html=True
)

# ---------------------- LOAD MODEL ----------------------
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="MobileNetV2_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ---------------------- FILE UPLOAD ----------------------
uploaded_file = st.file_uploader("📤 Choose a retinal image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Retinal Image", use_container_width=True)

    # Preprocess the image
    image = image.resize((224, 224))
    input_data = np.expand_dims(np.array(image, dtype=np.float32) / 255.0, axis=0)

    # Run model
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]

    # Define class names
    class_names = [
        "Diabetic Retinopathy",
        "Glaucoma",
        "Cataract",
        "Age-related Macular Degeneration",
        "Normal"
    ]

    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    # Show results
    st.success(f"✅ **Prediction:** {predicted_class}")
    st.info(f"🎯 **Confidence:** {confidence:.2f}%")

else:
    st.warning("👁️ Please upload a retinal image to begin diagnosis.")
