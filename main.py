from curses import A_ALTCHARSET
from re import A
from flask import Flask, render_template
import os
from flask import Flask, request, redirect, render_template, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import math
import pickle
import sklearn
import pandas
import sys
import tensorflow as tf


image_size_h = 157
image_size_w = 300

UPLOAD_FOLDER = './uploads/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.secret_key = "super secret key"


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

modelCTR = load_model('./Functional_CTR_model3.h5')
modelCVR = load_model('./Functional_CVR_model.h5')

@app.route('/', methods=['GET', 'POST'])

def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)

        files = request.files.getlist("file")
        if len(files) > 2:
            flash('最大2枚まで')
            return redirect(request.url)

        filepaths = []
        for file in files:
         if file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            filepath = os.path.join(UPLOAD_FOLDER, filename)

        imgs = []
        for filepath in filepaths:
            img = image.load_img(filepath, grayscale=False,
                        target_size=(image_size_h, image_size_w))
        imgs.append(image.img_to_array(img))
        imgs = np.array(imgs)


        baitailabele = request.form.get('baitai')
        ichilabel = request.form.get('ichi')
        baitailabele = int(baitailabele)
        ichilabel = int(ichilabel)
        baitai = []
        ichi = []
        for _ in range(len(imgs)):
            baitai.append([baitailabele])
            ichi.append([ichilabel])
        baitai = np.array(baitai)
        ichi = np.array(ichi)
        
        ctr_label = np.argmax(modelCTR.predict([baitai, ichi, imgs]))
        cvr_label = np.argmax(modelCVR.predict([baitai, ichi, imgs]))
        
        with open('./CTRle1.pickle', 'rb') as f1, open('./CVRle2.pickle', 'rb') as f2:
            le1 = pickle.load(f1)
            le2 = pickle.load(f2)
        ctrla = le1.inverse_transform([ctr_label])
        cvrla = le2.inverse_transform([cvr_label])
        answer = ""
        answer2 = ""
        for i in range(len(imgs)):
            answer += "{:.2f}%~{:.2f}% , {:.2f}%~{:.2f}, %".format((ctrla[i].left)*100,(ctrla[i].right)*100,(cvrla[i].left)*100,(cvrla[i].right)*100)
            clickleft = int((ctrla[i].left)*100000)
            clickright = int((ctrla[i].right)*100000)
            cvleft = int(clickleft * (cvrla[i].left))
            cvrright = int(clickleft * (cvrla[i].right))
            answer2 += "{}~{} , {}~{}, ".format(clickleft, clickright, cvleft, cvrright)

        return render_template("index.html", answer=answer, answer2=answer2)
    return render_template("index.html", answer="", answer2="")

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(debug = True,host ='0.0.0.0',port = port)
    port = int(os.environ.get('PORT', 8080))
    app.run(debug = True,host ='0.0.0.0',port = port)

