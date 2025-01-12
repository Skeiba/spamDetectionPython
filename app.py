from flask import Flask, request, jsonify, render_template
from joblib import load

model = load("model/spam_model.pkl")
vectorizer = load("model/tfidf_vectorizer.pkl")

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    email_message = request.json['email_message']
    vectorized_message = vectorizer.transform([email_message])
    prediction = model.predict(vectorized_message)
    
    result = "Spam" if prediction[0] == 1 else "Ham"
    return jsonify({"result": result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)