from flask import Flask

app = Flask(__name__)

print("print ok")


@app.route("/amzn", methods=["GET"])
def firstRoute():
    return "hello lakmal"


# app.register_blueprint("/l")

if __name__ == "__main__":
    app.run(port=5000)
