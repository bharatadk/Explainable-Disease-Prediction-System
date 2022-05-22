from flask import Flask, render_template, request,url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import pandas as pd
import numpy as np
import pickle
import time
from tb import *
from heart import *

app = Flask(__name__)

import warnings
warnings.filterwarnings('ignore')

@app.route("/")
def index():
	return render_template("home.html")

@app.route("/heart")
def heart():
	return render_template("heart.html")

@app.route("/tb")
def tb():
	return render_template("tb.html")


@app.route("/heart_predict", methods=['POST'])
def heart_predict():
	if request.method == 'POST':
		prediction_value = predict_heart()
	return render_template('heart_predict.html', prediction = prediction_value)


@app.route("/tb_predict", methods = ['POST'])
def get_output():

	try:
	
		if request.method == 'POST':
		
			img = request.files['my_image']
			img_path = "static/" + img.filename	
			img.save(img_path)
			p = predict_tb(img_path)
			saliency_map(img_path,check=p)
			time.sleep(1)
			
			return render_template("tb_predict.html", prediction = p, img_path = img_path)

	except Exception as e :

		print(e)
		return render_template('tb.html')
		



if __name__ =='__main__':
	app.debug = True
	app.run()