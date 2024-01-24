from app.data_processing.data_cleaning.data_cleaning_service import get_prediction
from fastapi import FastAPI, Query
from flask import Flask

app = Flask(__name__)

print("print ok")


@app.route("/get-prediction", methods=["GET"])
def firstRoute():
    # print("aa")
    # return "a"
    return get_prediction("TSLA", "2025-08-13")


# app.register_blueprint("/l")

if __name__ == "__main__":
    app.run(port=5000)
