import streamlit as st
from transformers import pipeline
from PIL import Image

st.set_page_config(page_title="Sentiment Analysis", layout="centered")

st.title("Sentiment Analysis")

# -------- LOAD MODELS --------
@st.cache_resource
def load_models():
    sentiment = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    
    caption = pipeline(
        "image-to-text",
        model="nlpconnect/vit-gpt2-image-captioning"
    )
    
    return sentiment, caption

sentiment_model, caption_model = load_models()

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
# IMAGE INPUT
# =========================
st.subheader("Image Input")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Analyze Image"):
        with st.spinner("Analyzing image..."):
            caption = caption_model(image)[0]["generated_text"]
            result = sentiment_model(caption)[0]

        st.write(f"Caption: {caption}")

        if result["label"] == "POSITIVE":
            st.success("Positive")
        else:
            st.error("Negative")

        st.write(f"Confidence: {round(result['score'], 3)}")