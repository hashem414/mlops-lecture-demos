from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open("model/model.pkl", "rb"))

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]
    prediction = model.predict(np.array(data).reshape(1, -1))
    return jsonify({"class": int(prediction[0])})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)