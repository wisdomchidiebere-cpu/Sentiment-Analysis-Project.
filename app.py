from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
# This allows your HTML file to talk to your Python server securely
CORS(app)

print("Loading the AI Brain...")
vectorizer = joblib.load('vectorizer.joblib')
model = joblib.load('model.joblib')
print("Brain loaded successfully!")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    
    # 1. Convert the new sentence into math
    vectorized_text = vectorizer.transform([text])
    
    # 2. Ask the model to predict (1 = Positive, 0 = Negative)
    prediction = model.predict(vectorized_text)[0]
    
    # 3. Translate the math back to English
    sentiment = "Positive 😊" if prediction == 1 else "Negative 😔"
    
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(port=5000)