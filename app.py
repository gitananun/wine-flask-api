import pandas as pd
from flask import Flask, request, jsonify
import pickle

app = Flask('Wine')
model = pickle.load(file := open('model.pkl', 'rb'))


@app.route('/', methods=['POST'])
def show():
    return jsonify({"data": request.json})


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    request.headers.add("Access-Control-Allow-Origin", "*")

    data_to_predict = [pd.Series(data)]

    prediction = model.predict(data_to_predict)

    output = round(prediction[0], 2)

    return jsonify({"data": f"The quality of your wine is {output} out of 10"})


file.close()

if __name__ == "__main__":
    app.run(debug=True)
