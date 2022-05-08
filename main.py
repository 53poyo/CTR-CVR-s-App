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



#モデルに入力される画像サイズを指定
image_size_h = 157
image_size_w = 300


#アップロードされた画像を保存するフォルダ名
# 画像のアップロード先のディレクトリも指定できる
UPLOAD_FOLDER = './uploads/'
#アップロードを許可する拡張子の制限
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

#appというFlaskクラスのインスタンスを作成します。app. < Flaskクラスのメソッド名 > とすることでFlaskクラスのメソッドを使えるようにする
app = Flask(__name__)
app.secret_key = "super secret key"


def allowed_file(filename):
    #filename小.文字の文字列が存在する&filenameの.より後ろの文字列が小文ALLOWED_EXTENSIONSのどれかに該当するか
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


modelCTR = load_model('./Functional_CTR_model3.h5')  # 学習済みモデルをロード
modelCVR = load_model('./Functional_CVR_model.h5')  # 学習済みモデルをロード

#app.route()は、次の行で定義される関数を指定した URL に対応づけるという処理をしています。
#GETでリソース(html)を取り込み、POSTでサーバーへ送信する


@app.route('/', methods=['GET', 'POST'])
#http://127.0.0.1:5000/ 以降のパスを指定する
#templateフォルダ内のindex.htmlを読み込む
#Flaskはtemplatesフォルダの中からhtmlファイルを探し、staticフォルダの中からcssファイルを探すことになっています
#def hello_world():
   #return render_template("index.html")
#request.methodにはリクエストのメソッドが格納#request.method == 'POST'のとき以降の処理が実行
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('ファイルがありません')
#redirect()は引数に与えられたurlに移動する
#request.url=リクエストがなされたページのURLが格納→、アップロードされたファイルがない、もしくはファイル名がない場合は元のページに戻る。
            return redirect(request.url)

#ファイルをリストとして取得する　filesのlen(枚数が２枚以下しか受け取らない)
        files = request.files.getlist("file")
        if len(files) > 2:
            flash('最大2枚まで')
            return redirect(request.url)

#[]送信されたデータを取得　名前を入れてないと送信しない
        filepaths = []
        for file in files:
        #file = request.files['file']
         if file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)

#secure_filenameでファイル名に危険な文字列がある場合に無効化
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            #filepathsのリストに複数枚入れる
            filepaths.append(filepath)
#=====================================================

#インデントは↑のifの中に入れない
#画像の処理１
#受け取った画像を読み込み、np形式に変換
    #image.load_imgでロードとリサイズ
        imgs = []
        for filepath in filepaths:
            img = image.load_img(filepath, grayscale=False,
                        target_size=(image_size_h, image_size_w))

        #image.img_to_arrayで読み込んだ画像をNumpy配列に変換する →imgsのリストに追加する
        imgs.append(image.img_to_array(img))
        #model.predict()にはNumpy配列のリストを渡す必要があるので格納する#複数予測したい時は、data = np.array([img,img...])
        #imgsのリストに格納されているimgをnp.arrayに変換する　 np.array([img,img...])の形にする
        img = np.array(imgs)


        #配信媒体と配信面の処理2（ラベル付）
        baitailabele = request.form.get('baitai')
        ichilabel = request.form.get('ichi')
        #print(type(baitailabele),file=sys.stderr)
        #htmlからの取得データは文字列のためint型に変換

        int(baitailabele)
        int(ichilabel)


        #配信媒体、配信面１×１の形、　画像：枚数（ひとつの画像だから１）×高さ×横×RGB　をリストにまとめる　モデルの順番に合わせる
        #複数予測したい時は、baitailabele = np.array([[baitailabele],[baitailabele]....])
        #こちらは、1x1のshapeにしたいので、([[baitailabele]])のように[]を二重にしています。[]が1つの場合は、shapeが(1,)で、2つの場合はshapeが(1, 1)になります。

        baitai = []
        ichi = []
        for _ in range(len(imgs)):
            baitai.append([baitailabele])
            ichi.append([ichi])

        baitai = np.array(baitai)
        ichi = np.array(ichi)

         #CTR ,CVRラベルを整数に戻す モデルの順番に合わせる　x1 =配信媒体　x2 =配信面　x3 = 画像　の順番に渡す。axis=1で横にリストを輪とめる
        ctr_label =[]
        cvr_label =[]
        for i in range(len(imgs)):
            ctr_label.append(np.argmax(modelCTR.predict([baitai[i], ichi[i], img[i]]), axis=1))
            cvr_label.append(np.argmax(modelCVR.predict([baitai[i], ichi[i], img[i]]), axis=1))
        #ctr_label = np.argmax(modelCTR.predict([baitai, ichi, img]), axis=1)
        #cvr_label = np.argmax(modelCVR.predict([baitai, ichi, img]), axis=1)



        #yの保存したラベルピッケルをここで開ける
        with open('./CTRle1.pickle', 'rb') as f1, open('./CVRle2.pickle', 'rb') as f2:
            le1 = pickle.load(f1)
            le2 = pickle.load(f2)

        # ↑開けたラベルピッケルの0, 1...の整数値を[0.1, 0.2]のような範囲に逆変換する
        ctrla = le1.inverse_transform(ctr_label)
        cvrla = le2.inverse_transform(cvr_label)

        #範囲の場合、pandasのレフト、ライトで取り出す。ctrlα、cvrlaの範囲を取り出している。 ctr[1]にしたら２つ目の値が出てくるのでフォーマットを分けて書く
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
#=====================================================
    return render_template("index.html", answer="", answer2="")



        #__name__ == '__main__'がTrueである、すなわちこのコードが直接実行されたときのみapp.run()が実行され、Flaskアプリが起動
#if __name__ == "__main__":
    #port = int(os.environ.get('PORT', 8080))
    #app.run(debug = True,host ='0.0.0.0',port = port)


#練習用
if __name__ == "__main__":
    app.run(debug = True, port = 5000)
