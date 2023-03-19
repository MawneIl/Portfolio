import dill

from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel


app = FastAPI()
with open('model/models/best_pipe.pkl', 'rb') as file:
    model = dill.load(file)


class Form(BaseModel):
    session_id: str
    utm_source: str | None = None
    utm_medium: str | None = None
    utm_campaign: str | None = None
    utm_keyword: str | None = None
    utm_adcontent: str | None = None
    device_category: str | None = None
    device_os: str | None = None
    device_brand: str | None = None
    device_model: str | None = None
    device_screen_resolution: str | None = None
    device_browser: str | None = None
    geo_country: str | None = None
    geo_city: str | None = None


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
        'session_id': form.session_id,
        'result': y[0]
    }


