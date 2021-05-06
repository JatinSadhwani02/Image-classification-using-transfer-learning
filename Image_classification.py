# Keras
# from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import numpy as np 
# from flask import Flask , redirect , url_for , request , render_template
# import os
# import glob
# from werkzeug.utils import secure_filename




from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)
model_path = 'Image_classification.h5'
model = load_model(model_path)
# model._make_predict_function()  # Necessory

def predictions(img_path , model):
    img =image.load_img(img_path , target_size=(224,224))

    x = image.img_to_array(img)
    x = np.expand_dims(x , axis=0)

    x = preprocess_input(x)

    preds = model.predict(x)

    return preds



@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict',methods=['GET','POST'])
def upload():
    if request.method=='POST':
        img = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath , 'uploads' , secure_filename(img.filename))
        img.save(file_path)

        pred = predictions(file_path , model) # return index
        pred_class = decode_predictions(pred , top=1) # map index with class label
        result = str(pred_class[0][0][1])

        return result
    return None 







app.run(debug=True)