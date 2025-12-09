import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# -------------------------------
# Helper: convert any image to 28x28 grayscale
# -------------------------------
def preprocess_image(img):
    img = img.convert("L")  # grayscale
    img = img.resize((28, 28))
    img_array = np.array(img)
    
    # عكس الألوان إذا الخلفية ساطعة
    if np.mean(img_array) > 127:
        img_array = 255 - img_array
    
    # تحويل الصورة إلى binary
    img_array = (img_array > 128).astype(np.float32)
    
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array


# -------------------------------
# Load model once without compile
# -------------------------------
model = load_model("best_model_cnn.keras", compile=False)

# -------------------------------
# Streamlit Page
# -------------------------------
st.title("Digit Prediction")

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
    pred = model.predict(processed, verbose=0)  # verbose=0 لتقليل المخرجات
    digit = int(np.argmax(pred))

    st.subheader(f"Predicted Digit: **{digit}**")

