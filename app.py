import pandas as pd
from flask import Flask, request, jsonify
import pickle

app = Flask('Wine')
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/', methods=['POST'])
def show():
    return jsonify(request.json)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    data_to_predict = [pd.Series(data)]

    prediction = model.predict(data_to_predict)

    output = round(prediction[0], 2)
    return f"The quality of your wine is {output}"


if __name__ == "__main__":
    app.run(debug=True)
