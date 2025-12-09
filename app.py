# streamlit_app.py
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# -------------------------------
# 1) Load trained model
# -------------------------------
@st.cache_resource
def load_cnn_model():
    return load_model("best_model_cnn.keras")

model = load_cnn_model()

# -------------------------------
# 2) App title
# -------------------------------
st.title("CNN Image Prediction")
st.write("Upload an image (28x28 grayscale or any size, will be resized)")

# -------------------------------
# 3) Upload image
# -------------------------------
uploaded_file = st.file_uploader("Choose an image...", type=["png","jpg","jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('L')  # convert to grayscale
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # Resize to 28x28 (or whatever input your CNN expects)
    img = img.resize((28,28))
    img_array = np.array(img).astype('float32') / 255.0
    img_array = img_array.reshape(1,28,28,1)  # add batch dimension
    
    # -------------------------------
    # 4) Make prediction
    # -------------------------------
    y_pred_prob = model.predict(img_array)
    y_pred = np.argmax(y_pred_prob, axis=1)[0]
    confidence = np.max(y_pred_prob)
    
    st.write(f"Predicted Class: **{y_pred}**")
    st.write(f"Confidence: **{confidence*100:.2f}%**")
