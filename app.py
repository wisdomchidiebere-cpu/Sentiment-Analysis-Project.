from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

print("Loading the AI Brain...")
vectorizer = joblib.load('vectorizer.joblib')
model = joblib.load('model.joblib')
print("Brain loaded successfully!")

# NEW: The Front Door! This serves your HTML page to the cloud.
@app.route('/')
def home():
    return send_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    
    vectorized_text = vectorizer.transform([text])
    prediction = model.predict(vectorized_text)[0]
    sentiment = "Positive 😊" if prediction == 1 else "Negative 😔"
    
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(port=5000)
