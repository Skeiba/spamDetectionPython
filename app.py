from flask import Flask,render_template, request, jsonify
import joblib

model = joblib.load('model/spam_model.pkl')
vectorizer = joblib.load('model/tfidf_vectorizer.pkl')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    email_text = request.json['text']
    vectorized_text = vectorizer.transform([email_text])
    prediction = model.predict(vectorized_text)
    result = 'spam' if prediction[0] == 1 else 'ham'
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)