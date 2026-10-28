from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

from embedding import embed, load_bert

# Loaded once when the server starts, not on every request.
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["tokenizer"], state["bert"] = load_bert()
    state["classifier"] = joblib.load("models/classifier.joblib")
    yield
    state.clear()


app = FastAPI(title="Portuguese fake news classifier", lifespan=lifespan)


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, description="News text in Portuguese")


class PredictResponse(BaseModel):
    label: str
    probability: float
    probabilities: dict[str, float]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    vector = embed([request.text], state["tokenizer"], state["bert"])
    classifier = state["classifier"]
    probabilities = classifier.predict_proba(vector)[0]
    probabilities = dict(zip(classifier.classes_, probabilities.tolist()))
    label = max(probabilities, key=probabilities.get)
    return PredictResponse(
        label=label,
        probability=probabilities[label],
        probabilities=probabilities,
    )
