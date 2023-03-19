import dill

from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel


app = FastAPI()
with open('model/models/best_pipe.pkl', 'rb') as file:
    model = dill.load(file)


class Form(BaseModel):
    session_id: str
    utm_source: str
    utm_medium: str
    utm_campaign: str
    utm_keyword: str
    device_category: str
    device_os: str
    device_brand: str
    device_model: str
    device_screen_resolution: str
    device_brand: str
    device_model: str
    device_screen_resolution: str
    device_browser: str
    geo_country: str
    geo_city: str


class Prediction(BaseModel):
    session_id: str
    result: float


@app.get('/status')
def status():
    return "I'm OK"


@app.get('/version')
def version():
    return model['metadata']

@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    df = pd.DataFrame.from_dict([form.dict()])
    y = model['model'].predict(df)

    return {
        'Loan_ID': form.session_id,
        'Result': y[0]
    }


