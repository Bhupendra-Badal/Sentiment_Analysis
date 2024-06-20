from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
import joblib

# Load the pre-trained model from a file (update the path to your model)
model = joblib.load("B:\Sentiment Analysis\logistic_regression_model.joblib")

# Define the request body model
class TextInput(BaseModel):
    texts: List[str]

# Initialize the FastAPI app
app = FastAPI()

# Mount the static directory for CSS
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_form():
    with open("templates/index.html", "r") as file:
        return HTMLResponse(content=file.read(), status_code=200)

@app.post("/predict-sentiment/")
async def predict_sentiment(input: TextInput):
    # Perform sentiment analysis using the loaded model
    predictions = [model.predict([text])[0] for text in input.texts]
    return {"predictions": predictions}
