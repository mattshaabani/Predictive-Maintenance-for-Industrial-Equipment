from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI()

model = joblib.load("models/random_forest_model.joblib")
preprocessor = joblib.load("models/preprocessor.joblib")

class PredictionInput(BaseModel):
    Type: str
    Air_temperature: float
    Process_temperature: float
    Rotational_speed: float
    Torque: float
    Tool_wear: float

@app.post("/predict")
async def predict(input_data: PredictionInput):
    try:
        data = {
            "Type": [input_data.Type],
            "Air temperature [K]": [input_data.Air_temperature],
            "Process temperature [K]": [input_data.Process_temperature],
            "Rotational speed [rpm]": [input_data.Rotational_speed],
            "Torque [Nm]": [input_data.Torque],
            "Tool wear [min]": [input_data.Tool_wear]
        }
        df = pd.DataFrame(data)

        X = preprocessor.transform(df)

        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0].tolist()

        return {
            "prediction": int(prediction),
            "probability": probability
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Predictive Maintenance API"}