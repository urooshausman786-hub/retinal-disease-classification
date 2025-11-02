import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Retinal Disease Classifier",
    page_icon="🩺",
    layout="centered"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .stApp {
        background-color: #f4f6fa;
        font-family: 'Poppins', sans-serif;
    }
    .main-title {
        color: #1b263b;
        text-align: center;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        color: #415a77;
        text-align: center;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .stButton>button {
        background-color: #1b263b;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-weight: 600;
        border: none;
    }
    .stButton>button:hover {
        background-color: #0d1b2a;
        color: #e0e1dd;
    }
</style>
""", unsafe_allow_html=True)

# --- PAGE HEADER ---
st.markdown('<p class="main-title">🩺 Retinal Disease Classification</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Upload a retinal image to detect possible eye diseases using AI</p>', unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="MobileNetV2_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --- FILE UPLOADER ---
uploaded_file = st.file_uploader("📤 Choose a retinal image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🩻 Uploaded Retinal Image", use_container_width=True)

    with st.spinner("🔍 Analyzing the image... Please wait"):
        image_resized = image.resize((224, 224))
        input_data = np.expand_dims(np.array(image_resized, dtype=np.float32) / 255.0, axis=0)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]

    # --- CLASSIFICATION ---
    class_names = [
        "Diabetic Retinopathy",
        "Glaucoma",
        "Cataract",
        "Age-related Macular Degeneration",
        "Normal"
    ]

    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    st.success(f"🎯 **Prediction:** {predicted_class}")
    st.info(f"📊 **Confidence:** {confidence:.2f}%")

else:
    st.warning("👆 Please upload an image to begin diagnosis.")

# --- FOOTER ---
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>Made with ❤️ by <b>Uroosha Usman</b> | MSc Computer Science</p>",
    unsafe_allow_html=True
)
