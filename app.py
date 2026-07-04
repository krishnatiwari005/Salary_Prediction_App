from flask import Flask, jsonify, request,render_template

import joblib

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route('/')

def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])

def predict():
    experience = float(request.form['experience'])
    prediction = model.predict([[experience]])
    return render_template('index.html', prediction_text=f'Predicted Salary: {prediction[0]:,.2f}')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)