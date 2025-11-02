import streamlit as st
import numpy as np
from PIL import Image
import tensorflow.lite as tflite


st.set_page_config(page_title="Retinal Disease Classifier", page_icon="🩺")

st.title("🩺 Retinal Disease Classification")
st.write("Upload a retinal image to predict the possible disease using a TFLite model.")

@st.cache_resource
def load_model():
    try:
        interpreter = tflite.Interpreter(model_path="MobileNetV2_model.tflite")
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

uploaded_file = st.file_uploader("Choose a retinal image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    image = image.resize((224, 224))
    input_data = np.expand_dims(np.array(image, dtype=np.float32) / 255.0, axis=0)

    # Run inference
    try:
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    except Exception as e:
        st.error(f"Inference error: {e}")
        st.stop()

    # Labels (edit as per your dataset)
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
