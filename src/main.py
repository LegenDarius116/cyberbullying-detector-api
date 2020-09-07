from typing import Optional
from fastapi import FastAPI
from joblib import load


text_classifier = load("model/model_v3.model")
app = FastAPI()


@app.get("/analyze/")
async def analyze_text(q: Optional[str] = None):
    """Takes the query string, and then uses a text classifier to analyze it for cyberbullying content.

    This will return the text, a flag indicating if it is cyberbullying (true) or not, and a confidence rating
    between 0.0 and 1.0 on that prediction.

    - **q**: Input string to analyze
    """
    text = q
    is_cyberbullying = False
    probability = 0.0

    if q is not None:
        is_cyberbullying = text_classifier.predict([text])[0]
        probability = text_classifier.predict_proba([text])[0][1]

    return {
        "text": text,
        "is_cyberbullying": is_cyberbullying,
        "probability": probability
    }
