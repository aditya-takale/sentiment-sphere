import streamlit as st
import requests

st.set_page_config(
    page_title="Sentiment Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- SIDEBAR ----------
with st.sidebar:
    st.title("Sentiment App")
    st.write("Analyze text sentiment using AI")
    st.divider()
    st.write("Status: Running")
    st.write("Model: DistilBERT")

# ---------- MAIN TITLE ----------
st.title("Sentiment Analysis")

st.write("Enter text below to analyze sentiment.")

# ---------- MAIN LAYOUT ----------
col1, col2 = st.columns([2, 1])

# ---------- INPUT SECTION ----------
with col1:
    st.subheader("Input")

    user_input = st.text_area(
        "Text",
        height=150,
        placeholder="Type something..."
    )

    if st.button("Analyze"):
        if user_input.strip():
            try:
                res = requests.post(
                    "http://127.0.0.1:8000/predict",
                    json={"text": user_input}
                )
                result = res.json()

                st.subheader("Result")

                if result["label"] == "POSITIVE":
                    st.success("Positive")
                else:
                    st.error("Negative")

                st.write(f"Confidence: {result['score']}")

            except:
                st.error("API not running")

        else:
            st.warning("Enter text first")

# ---------- INFO PANEL ----------
with col2:
    st.subheader("System Info")
    st.write("Model: DistilBERT")
    st.write("API: Connected")
    st.write("Status: Active")