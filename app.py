from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load the trained model
with open("iris_rf_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Iris Classifier API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = np.array(data["features"]).reshape(1, -1)

        # Validate input size
        if len(features[0]) != 4:
            return jsonify({"error": "Invalid input size, must be 4 features"}), 400

        prediction = model.predict(features)

        return jsonify({"prediction": prediction[0]})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
