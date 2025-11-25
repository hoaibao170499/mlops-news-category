from pydantic import BaseModel

class NewsPredictionResponse(BaseModel):
    predicted_news: str