# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

model = joblib.load("exp-sal.pkl")


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['Get', 'POST'])
def predict():
    global df
    
    input_features = [float(x) for x in request.form.values()]
    features_value = np.array(input_features)
    
    
    
    # Validating input hours
    
    if input_features[0] == 0:
        return render_template('index.html', prediction_text='Please enter valid years of experience')
        

    output = model.predict([features_value])[0].round(2)

    
    return render_template('index.html', 
                           prediction_text='For {} years of experience, the salary will be {}'.format(input_features[0],output))
                           


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
    