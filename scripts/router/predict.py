import os

import mlflow.sklearn
import pandas as pd
from fastapi import APIRouter

from scripts.schemas.request import NewsPredictionRequest
from scripts.schemas.response import NewsPredictionResponse

MLFLOW_TRACKING_URI = os.getenv("OUR_MLFLOW_HOST", "http://localhost:5050")
print(f"MLFLOW_TRACKING_URI: {MLFLOW_TRACKING_URI}")

mlflow.set_tracking_uri(uri=MLFLOW_TRACKING_URI)

model_name = "news_classifier"
model_version = "1"
alias = "the_best"

model_uri = f"models:/{model_name}/{model_version}"

model = mlflow.sklearn.load_model(model_uri)

news_router = APIRouter(prefix="/news")

ID_TO_LABEL = {
    0: 'atheism',
    1: 'religion.misc',
    2: 'graphics',
    3: 'space',
}

# /news/predict
@news_router.post("/predict", response_model=NewsPredictionResponse)
def func_predict(request: NewsPredictionRequest) -> NewsPredictionResponse:
    input_data = {
        "posts": [request.posts],
    }   
    df = pd.DataFrame(input_data)
    predictions = model.predict(df['posts'])
    predicted_labels = [ID_TO_LABEL.get(int(p), str(p)) for p in predictions]
    return NewsPredictionResponse(predicted_news=predicted_labels[0])