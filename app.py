import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os

# Set page config
st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🫁",
    layout="centered"
)


# Load model
@st.cache_resource
def load_cnn_model():
    model_path = "model.keras"
    if not os.path.exists(model_path):
        st.error(f"❌ Model file '{model_path}' not found!")
        st.error("📁 Current directory contents:")
        st.write(os.listdir('.'))
        st.info("💡 Make sure 'model.keras' is uploaded to your repository")
        st.stop()

    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()


# Load model
model = load_cnn_model()

# Header
st.title("🩺 Pneumonia Detection from Chest X-ray")
st.markdown("""
Upload a chest X-ray image and our AI model will analyze it to detect signs of pneumonia.
""")

# Add info about the model
with st.expander("ℹ️ About this Model"):
    st.markdown("""
    - **Architecture**: MobileNetV2 with custom classification head
    - **Input Size**: 128x128 pixels
    - **Classes**: Normal vs Pneumonia
    """)

# File uploader
uploaded_file = st.file_uploader(
    "📤 Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG"
)

if uploaded_file is not None:
    try:
        # Load and display image
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="📸 Uploaded X-ray", width=300)

        with col2:
            # Preprocess image
            st.write("🔄 **Processing Steps:**")
            st.write("1. ✅ Image loaded")

            # Resize to model input size
            img_resized = image.resize((128, 128))

            # Convert to array and normalize
            img_array = img_to_array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            st.write("3. ✅ Normalized pixel values")

            # Make prediction
            st.write("4. 🧠 Making prediction...")

        # Predict
        with st.spinner('🔍 Analyzing X-ray...'):
            prediction = model.predict(img_array, verbose=0)[0][0]

        # Display results
        st.markdown("---")
        st.subheader("📊 Analysis Results")

        col1, col2, col3 = st.columns(3)

        with col2:  # Center column
            if prediction >= 0.5:
                confidence = prediction * 100
                st.error(f"🚨 **PNEUMONIA DETECTED**")
                st.error(f"**Confidence: {confidence:.1f}%**")
                st.markdown("⚠️ *Please consult a medical professional*")
            else:
                confidence = (1 - prediction) * 100
                st.success(f"✅ **NORMAL**")
                st.success(f"**Confidence: {confidence:.1f}%**")
                st.markdown("😊 *No signs of pneumonia detected*")

        # Show confidence bar
        st.markdown("### 📈 Confidence Levels")
        normal_conf = (1 - prediction) * 100
        pneumonia_conf = prediction * 100

        st.write("**Normal:**")
        st.progress(normal_conf / 100)
        st.write(f"{normal_conf:.1f}%")

        st.write("**Pneumonia:**")
        st.progress(pneumonia_conf / 100)
        st.write(f"{pneumonia_conf:.1f}%")

    except Exception as e:
        st.write(' ')
        
# Sidebar info
with st.sidebar:
    st.markdown("### 🔬 How it works")
    st.markdown("""
    1. **Upload** a chest X-ray image
    2. **AI analyzes** the image using deep learning
    3. **Results** show probability of pneumonia
    4. **Confidence** scores help interpret results
    """)

    st.markdown("### 📋 Tips for best results")
    st.markdown("""
    - Use clear, high-quality X-ray images
    - Ensure the image shows the full chest area
    - Standard chest X-ray positioning works best
    """)

# Footer disclaimer
st.markdown("---")
st.markdown("""
### ⚠️ **Medical Disclaimer**

**IMPORTANT**: This application is for **educational and research purposes only**. 

- 🚫 **NOT for medical diagnosis**
- 🏥 **Always consult healthcare professionals**
- 📚 **For learning purposes only**
- ⚖️ **Not FDA approved**

**Never use this tool as a substitute for professional medical advice, diagnosis, or treatment.**
""")

st.markdown("---")
st.markdown("*Made with ❤️ for medical AI education*")