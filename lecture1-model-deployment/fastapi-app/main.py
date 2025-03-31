from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI()
model = pickle.load(open("../model/model.pkl", "rb"))

@app.post("/predict")
async def predict(features: list):
    prediction = model.predict(np.array(features).reshape(1, -1))
    return {"class": int(prediction[0])}
