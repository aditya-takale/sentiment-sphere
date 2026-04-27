import streamlit as st
from transformers import pipeline
from PIL import Image

st.set_page_config(page_title="Sentiment Analysis", layout="centered")

st.title("Sentiment Analysis")

# -------- LOAD MODEL --------
@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

sentiment_model = load_model()

# =========================
# TEXT INPUT
# =========================
st.subheader("Text Input")

text = st.text_area("Enter text")

if st.button("Analyze Text"):
    if text.strip():
        result = sentiment_model(text)[0]

        if result["label"] == "POSITIVE":
            st.success("Positive")
        else:
            st.error("Negative")

        st.write(f"Confidence: {round(result['score'], 3)}")
    else:
        st.warning("Enter text")

# =========================
# IMAGE INPUT (SAFE VERSION)
# =========================
st.subheader("Image Input")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.info(
        "⚠️ Image sentiment analysis is available in the local version. "
        "This cloud version shows image preview due to resource limitations."
    )