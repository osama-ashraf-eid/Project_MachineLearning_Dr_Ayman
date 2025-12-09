import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# -------------------------------
# Helper: convert any image to 28x28 grayscale
# -------------------------------
def preprocess_image(img):
    img = img.convert("L")                 # convert to grayscale
    img = img.resize((28, 28))             # resize
    img_array = np.array(img) / 255.0      # normalize
    img_array = img_array.reshape(1, 28, 28, 1)  # reshape for model
    return img_array

# -------------------------------
# Load model once and cache it
# -------------------------------
@st.cache_resource
def load_cnn_model():
    return load_model("best_model_cnn.keras")

model = load_cnn_model()

# -------------------------------
# Streamlit Page
# -------------------------------
st.title("Prediction")

# صورة مالـيّة العرض (fit)
st.image(
    "https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png",
    use_column_width=True
)

st.write("Upload a digit image and the model will predict a number from 0 to 9")

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Your Image", use_column_width=True)

    processed = preprocess_image(img)
    pred = model.predict(processed, verbose=0)  # امن لو رفعنا صور كتير
    digit = np.argmax(pred)

    st.subheader(f"Predicted Digit: **{digit}**")
