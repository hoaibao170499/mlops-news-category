import uvicorn
from fastapi import FastAPI
from scripts.router import predict

app = FastAPI()
app.include_router(predict.news_router)

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=3000, reload=True)