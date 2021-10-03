# -*- coding: utf-8 -*-
from flask import Flask,render_template
import joblib
from flask import request

import json
from flask import jsonify
import numpy as np
import os


#http://localhost:8080/predict?v2=23.2&v4=42.1&v11=52.42&amount=124.21

app = Flask(__name__)
#app = Api(app)
app.config['JSON_SORT_KEYS'] = False



@app.route('/predict',methods=['GET'])
def predict():
        dir_path = os.path.dirname(os.path.realpath(__file__))
        try:
             clf = joblib.load("pred_model.pkl".format(dir_path))
        except:
            return jsonify({'error': 'Failed to load model'})
    
        V2 = request.args.get('v2')
        V4 = request.args.get('v4')
        V11 = request.args.get('v11')
        Amount = request.args.get('amount')
       
        prediction = clf.predict([[V2,V4,V11,Amount]])
        
        return jsonify({"Classification":str(prediction[0])})
     
@app.route('/')
def home():
    return render_template('index.html')


if __name__ == "__main__":
        app.run(threaded=True,port=8080)