# -*- coding: utf-8 -*-
import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)


@app.route('/predict', methods=['GET'])
def predict():
    
        try:
             model = joblib.load('model.pkl')
        except:
            return jsonify({'error': 'Failed to load model'})
    
        V2 = request.args.get['v2']
        V4 = request.args.get['v4']
        V11 = request.args.get['v11']
        Amount = request.args.get['amount']
       
        prediction = model.predict([[V2,V4,V11,Amount]])

        return jsonify({str(prediction[0])})
     
@app.route('/')
def home():
    return render_template('index.html')

if __name__ == "__main__":
   app.run(threaded=True,port=8080)