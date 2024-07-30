from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load('linear_svc_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    transformed_review = vectorizer.transform([review]).toarray()
    prediction = model.predict(transformed_review)
    result = 'Fake Review' if prediction[0] == 1 else 'Genuine Review'
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
