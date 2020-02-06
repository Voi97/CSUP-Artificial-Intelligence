import os
import sys
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from keras.applications import MobileNetV2
import numpy as np
import cv2
from util import base64_to_pil

app = Flask (__name__)

MODEL_PATH = 'models/cp-0010.h5'
CATEGORIES = ["Dog", "Cat"]
model = load_model(MODEL_PATH)
model._make_predict_function()

def model_predict(img,model):


   IMG_RESIZE = 244
   x = cv2.cvtColor(np.array(img), cv2.IMREAD_GRAYSCALE)
   xx = cv2.resize(x,(IMG_RESIZE,IMG_RESIZE))
   xxx = xx.reshape(-1,IMG_RESIZE,IMG_RESIZE,1)
   preds = model.predict(xxx)
   return preds





@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        img = base64_to_pil(request.json)
        preds = model_predict(img,model)

        pred_probability = "{:.3f}".format(np.amax(preds))

        y = np.where(preds == np.amax(preds))
        yy = np.amax (y)
        result = CATEGORIES[int(preds[0][0])]
        print(preds)
        print(y)
        print(yy)
        print(np.amax(preds))
        print(preds)

        return jsonify(result=result, probability=pred_probability)
    return None

if __name__ == '__main__':
    http_server = WSGIServer(('0.0.0.0',5000),app)
    http_server.serve_forever()