# app.py
"""
Simple Streamlit app with 3 pages: Home, Analysis, Prediction.
Home and Analysis are placeholders.
Prediction page:
 - loads best_model_cnn.keras from same folder
 - accepts single or multiple images
 - converts ANY image to 28x28x1 (grayscale, normalized)
 - predicts using the loaded model and shows results
"""

import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import pandas as pd
import os

st.set_page_config(page_title="Digit Classifier", layout="wide")

# ---------------------
# Helpers
# ---------------------
@st.cache_resource(show_spinner=False)
def load_cnn_model(path="best_model_cnn.keras"):
    if not os.path.exists(path):
        return None
    return tf.keras.models.load_model(path)

def to_28x28_gray_array(pil_image):
    """
    Convert a PIL image (any mode/size) to a numpy array shaped (1,28,28,1),
    dtype float32, normalized to [0,1].
    """
    img = pil_image.convert("L")  # grayscale
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    arr = np.array(img).astype("float32") / 255.0
    arr = arr.reshape(1, 28, 28, 1)
    return arr

def preprocess_batch(pil_images):
    """
    Take a list of PIL images and return a numpy array shaped (N,28,28,1)
    """
    processed = [to_28x28_gray_array(img) for img in pil_images]
    batch = np.vstack(processed)  # result shape: (N,28,28,1)
    return batch

def predict_batch(model, pil_images):
    """
    Returns labels (list of ints) and confidences (list of floats)
    """
    if model is None:
        raise ValueError("Model is not loaded")
    X = preprocess_batch(pil_images)
    preds = model.predict(X)
    if preds.ndim == 1:
        # single-output case
        labels = [float(x) for x in preds]
        confs = [1.0 for _ in preds]
    else:
        # treat outputs as logits or probs for classes
        probs = tf.nn.softmax(preds, axis=1).numpy()
        labels = list(np.argmax(probs, axis=1).astype(int))
        confs = list(np.max(probs, axis=1).astype(float))
    return labels, confs

# ---------------------
# Page functions (placeholders for Home & Analysis)
# ---------------------
def home_page():
    st.title("Home")
    st.write("")  # placeholder

def analysis_page():
    st.title("Analysis")
    st.write("")  # placeholder

def prediction_page():
    st.title("Prediction")
    st.markdown("Upload one or more images. Each image will be converted to 28×28 grayscale before prediction.")
    st.info("Place the trained model file `best_model_cnn.keras` in the same folder as this app.")

    # load model
    model = load_cnn_model()
    if model is None:
        st.error("Model file `best_model_cnn.keras` not found or failed to load.")
        st.stop()
    else:
        st.success("Model loaded successfully.")
        try:
            st.caption(f"Model input shape: {model.input_shape}")
        except Exception:
            pass

    uploaded = st.file_uploader("Select image files (png/jpg/jpeg)", type=["png","jpg","jpeg"], accept_multiple_files=True)
    if not uploaded:
        st.info("No files uploaded yet.")
        return

    pil_images = []
    filenames = []
    for f in uploaded:
        try:
            img = Image.open(io.BytesIO(f.read()))
            pil_images.append(img.copy())
            filenames.append(f.name)
        except Exception as e:
            st.warning(f"Failed to load {getattr(f, 'name', 'file')}: {e}")

    if len(pil_images) == 0:
        st.warning("No valid images to process.")
        return

    # Predict
    with st.spinner("Preprocessing and predicting..."):
        try:
            labels, confs = predict_batch(model, pil_images)
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            return

    # Display results (grid of 3 columns)
    st.subheader("Results")
    cols = st.columns(3)
    for i, (img, name, lbl, conf) in enumerate(zip(pil_images, filenames, labels, confs)):
        col = cols[i % 3]
        with col:
            st.image(img, use_column_width=True, caption=f"{name}\nPred: {lbl}  (conf: {conf:.2f})")

    # Summary dataframe
    df = pd.DataFrame({
        "filename": filenames,
        "prediction": labels,
        "confidence": confs
    })
    st.subheader("Summary")
    st.dataframe(df, use_container_width=True)

    # Download CSV
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download results CSV", data=csv, file_name="predictions.csv", mime="text/csv")

# ---------------------
# Navigation & run
# ---------------------
page = st.sidebar.selectbox("Choose page", ["Home", "Analysis", "Prediction"])
if page == "Home":
    home_page()
elif page == "Analysis":
    analysis_page()
elif page == "Prediction":
    prediction_page()
