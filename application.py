
import pickle
from flask import Flask,render_template,jsonify, request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

## import ridge regressor and standard scaler
ridge_model = pickle.load(open('model/ridge.pkl','rb'))
StandardScaler = pickle.load(open('model/scaler.pkl','rb'))

 
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_data',methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'POST':
        Temprature=float(request.form.get('Temprature'))
        RH=float(request.form.get('RH'))
        Ws=float(request.form.get('Ws'))
        Rain=float(request.form.get('Rain'))
        FFMC=float(request.form.get('FFMC'))
        DMC=float(request.form.get('DMC'))
        ISI=float(request.form.get('ISI'))
        Classes=float(request.form.get('Classes'))
        Region=float(request.form.get('Region'))

        input_data = np.array([[Temprature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        input_scaled=StandardScaler.transform(input_data)
        result=ridge_model.predict(input_scaled)

        return render_template('home.html',result=result[0])
    else:
        return render_template('home.html')
        
    
if __name__ == '__main__':
    app.run(host="0.0.0.0")
