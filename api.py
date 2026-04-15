from fastapi import FastAPI
from pydantic import BaseModel
from model import predict_sentiment
from db import save_result

app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "🚀 Sentiment API is running"}

@app.post("/predict")
def get_sentiment(input: TextInput):
    text = input.text

    result = predict_sentiment(text)

    data = {
        "text": text,
        "label": result["label"],
        "score": result["score"]
    }

    save_result(data)  # safe now

    return result