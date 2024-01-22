from app.data_processing.data_cleaning.step_four import test_function
from flask import Flask

app = Flask(__name__)

print("print ok")


@app.route("/amzn", methods=["GET"])
def firstRoute():
    # print("aa")
    # return "a"
    return test_function()


# app.register_blueprint("/l")

if __name__ == "__main__":
    app.run(port=5000)
