import pandas as pd
from flask import Flask, request, jsonify
import pickle

app = Flask("Wine")
model = pickle.load(file := open("model.pkl", "rb"))


@app.route("/", methods=["POST"])
def show():
    return jsonify({"data": request.json})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    data_to_predict = [pd.Series(data)]

    prediction = model.predict(data_to_predict)

    output = round(prediction[0], 2)

    output = 10 if output > 10 else output
    output = 0 if output < 0 else output

    return jsonify({"value": output, "max": 10})


file.close()

if __name__ == "__main__":
    app.run(debug=True)
