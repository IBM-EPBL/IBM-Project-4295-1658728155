import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask,render_template,request
import cv2


my_app=Flask(__name__)

model=load_model("RTCS_classifier.h5")

@my_app.route('/')
def index():
    return render_template('index.html')


@my_app.route('/submit',methods=['GET','POST'])
def upload():
    if request.method=='POST':
        f=request.files['my_image'] 
        basepath=os.path.dirname(__file__)
        filepath=os.path.join(basepath,'uploads',f.filename)        
        f.save(filepath)
       # img = image.load_img(filepath)
        img = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,(128,128),1)
        img=np.expand_dims(img,axis=0)
        pred=model.predict(img)
        labels_dict = {0:'0', 1:'A',  2:'B',  3:'C',  4:'D',  5:'E', 6:'F', 7:'G', 8:'H', 9:'I',
                 10:'J', 11:'K', 12:'L', 13:'M', 14:'N', 15:'O', 16:'P',
                 17:"Q", 18:'R', 19:'S', 20:'T',  21:'U',  22:'V',  23:'W', 24:'X', 25:'Y', 26:'Z'}
        key = np.argmax(pred)
        ans = labels_dict.get(key)
        text=str(ans)      
        #file:///C:/Users/jessi/Real_Time_Communication_System/uploads/q1.png
        img_path = 'uploads/'+ f.filename
        id=True
    #return text
    return render_template("index.html", prediction = text, img_path = img_path,id=id)

if __name__=="__main__":
    my_app.run(port=5000,debug=True)