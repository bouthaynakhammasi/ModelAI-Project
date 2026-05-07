import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Dermate Skin Disease Recognition", layout="centered")

st.title("Dermate: Skin Disease Recognition")
st.write("Upload an image of the skin condition to get a prediction.")

@st.cache_resource
def load_resources():
    interp = tf.lite.Interpreter(model_path="./converted_model.tflite")
    interp.allocate_tensors()
    with open("labels.txt", "r") as f:
        lbls = [line.strip() for line in f.readlines()]
    return interp, lbls

try:
    interpreter, labels = load_resources()
except Exception as e:
    st.error(f"Error loading model: {e}")
    interpreter, labels = None, []

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and interpreter is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Predicting...")

    arr = np.expand_dims(np.array(image.resize((224, 224)), dtype=np.uint8), axis=0)

    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    interpreter.set_tensor(input_index, arr)
    interpreter.invoke()
    preds_raw = interpreter.get_tensor(output_index)[0]

    out_details = interpreter.get_output_details()[0]
    if out_details["dtype"] == np.uint8:
        scale, zero_point = out_details["quantization"]
        preds = (preds_raw.astype(np.float32) - zero_point) * scale
    else:
        preds = preds_raw.astype(np.float32)

    predicted_class = int(np.argmax(preds))
    confidence = float(np.max(preds))

    st.success(f"Prediction: **{labels[predicted_class]}**")
    st.info(f"Confidence: {confidence:.2%}")

    st.write("### Top 3:")
    top3 = sorted(enumerate(preds), key=lambda x: -x[1])[:3]
    for i, p in top3:
        st.write(f"- **{labels[i]}** : {p*100:.2f}%")
