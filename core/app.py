from app.data_processing.data_cleaning.data_cleaning_service import get_prediction
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from flask import Flask
from pydantic import BaseModel

app = FastAPI()

print("print ok")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any origin
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


class InputData(BaseModel):
    crypto_name: str
    date: str


@app.get("/get-prediction")
def firstRoute(
    crypto_name: str = Query(..., alias="crypto_name"),
    date: str = Query(..., alias="date"),
):
    print(f"Crypto Name: {crypto_name}, Date: {date}")
    return get_prediction("BTC", "2025-08-13")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
