#!/usr/bin/env python
# coding: utf-8

from flask import Flask, request, jsonify, render_template
import pickle


model = pickle.load(open('linear_regression_model.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    bmi = float(request.form['bmi'])
    physical_health_days = float(request.form['physical_health_days'])

    mental_health_days = model.predict([[bmi, physical_health_days]])

    return jsonify({'mental_health_days': mental_health_days[0]})


if __name__ == '__main__':
    app.run(debug=True)

