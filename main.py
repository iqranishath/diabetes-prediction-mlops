from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Diabetes Prediction API")
model = joblib.load("diabetes_model.pkl")


class DiabetesInput(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    BMI: float
    Age: int


@app.get("/")
def root():
    return {"message": "Diabetes Prediction API is live"}


@app.post("/predict")
def predict(data: DiabetesInput):
    X = pd.DataFrame([data.model_dump()])
    return {"diabetic": bool(model.predict(X)[0])}