from pydantic import BaseModel

class NewsPredictionRequest(BaseModel):
    posts: str