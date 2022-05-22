import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import pickle
import shap

import time
import os
from flask import request

def predict_heart():

    open_model = open('MODELS/heart_model.pkl','rb')
    heart_model = pickle.load(open_model)

    x_train = None
    x_train = pd.read_csv('DATASETS/heart.csv').tail().drop('target',axis = 1)
    X5 = shap.utils.sample(x_train, 5)
    explainer = shap.Explainer(heart_model.predict, X5)

    age =               float(request.form['age'])
    sex =               float(request.form['sex'])
    chest_pain_type =   float(request.form['cp'])
    resting_bps =       float(request.form['bps'])
    cholestrol =        float(request.form['ch'])
    fbs=                float(request.form['fbs'])
    resting_ecg =       float(request.form['ecg'])
    max_heart_rate =    float(request.form['mhr'])
    exercise_angina =   float(request.form['ex'])
    oldpeak =           float(request.form['op'])
    st_slope =          float(request.form['slp'])


    pred_args = [[age,sex,chest_pain_type,resting_bps,cholestrol,fbs,resting_ecg,max_heart_rate,exercise_angina,oldpeak,st_slope]]
    df = pd.DataFrame(pred_args,columns=['age','sex','cp','bps','ch','fbs','ecg','mhr','ex','op','slp'])
    model_predcition = heart_model.predict(df)

    x_test=df
    shap_values = explainer(x_test[0:1])
    matplotlib.use('Agg')
    fig=plt.gcf()
    time.sleep(1)
    shap.plots.waterfall(shap_values[0],max_display=11)
    plt.close()
    fig.savefig('static/shap_img.svg',dpi=300)
    time.sleep(1)

    res = "some default value to avoid error"
    pred_args=[]

    if model_predcition == 1:
        res = 'Affected'
    else:
        res = 'Not affected' 
    return res   