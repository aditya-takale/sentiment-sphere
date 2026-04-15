import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Sentiment Analysis")

st.title("Sentiment Analysis")

@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

model = load_model()

text = st.text_area("Enter text")

if st.button("Analyze"):
    if text.strip():
        result = model(text)[0]

        if result["label"] == "POSITIVE":
            st.success("Positive")
        else:
            st.error("Negative")

        st.write(f"Confidence: {round(result['score'], 3)}")
    else:
        st.warning("Enter text")