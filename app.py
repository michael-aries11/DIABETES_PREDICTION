import flask
import numpy as np
import pandas as pd
import pickle
from flask import Flask, render_template, request

app = Flask('Diabetes_Predictor')

@app.route('/')
def show_predict_diabetes_form():
    return render_template('predictForm.html')

@app.route('/result', methods=['POST'])
def results():
    form = request.form
    if request.method == 'POST':
      #write your function that loads the model
        model = pickle.load(open('./logreg_deployed.pkl', 'rb'))


        # Initialise data to lists. 
        data = [{'age': request.form['age'],
                 'al': request.form['al'],
                 'su':request.form['su'],
                 'rbc': request.form['rbc'],
                 'pc' : request.form['pc'],
                 'bgr': request.form['bgr'],
                 'bu' : request.form['bu'],
                 'htn' : request.form['htn'],
                 'pe': request.form['pe'],                
                 'class' : request.form['class']
                 }] 
      

        #   Creates DataFrame. 
        df = pd.DataFrame(data) 
        labels =['age', 'al', 'su', 'rbc', 'pc', 'bgr', 'bu', 'htn', 'pe', 'class']
        df = df[labels]
                
       
        predicted_diabetes_prevalence = model.predict(df)
        
    return render_template('results.html', predicted_result=predicted_diabetes_prevalence[0])

app.run("localhost", "9999", debug=True)