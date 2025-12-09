import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# تحميل الموديل
model = tf.keras.models.load_model("best_model_cnn.keras")

# ---- Title in center ----
st.markdown(
    "<h1 style='text-align: center;'>Handwritten Digit Recognition</h1>",
    unsafe_allow_html=True
)

# ---- Image under title ----
st.image(
    "https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png",
    caption="Example of handwritten digits",
    use_column_width=True
)

st.write("---")

st.write("Upload an image of a digit (any size or color)")

# رفع الصورة
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocessing
    image = image.convert("L")
    image = image.resize((28, 28))
    image = np.array(image) / 255.0
    image = image.reshape(1, 28, 28, 1)

    # Prediction
    prediction = model.predict(image)
    predicted_digit = np.argmax(prediction)

    st.success(f"✅ Predicted Digit: **{predicted_digit}**")
