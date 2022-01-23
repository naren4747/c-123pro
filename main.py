from tkinter import E
from webbrowser import get
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os,ssl,time

X=np.load("image.npz")['arr_0']
y=pd.read_csv("labels.csv")["labels"]

x_train,x_test,y_train,y_test = train_test_split(X,y,random_state=0,test_size=0.25)

x_train_scaled=x_train
x_test_scaled=x_test

clasifier=LogisticRegression(solver="saga",multi_class="multinomial").fit(x_train_scaled,y_train)

yprediction=clasifier.predict(x_test_scaled)

accuracy=accuracy_score(y_test,yprediction)
print(accuracy)

cap=cv2.VideoCapture(0)

while(True) :
    try:    
        ret,frame=cap.read()

        grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        height,width=grey.shape
        upperLeft=(int(width/2-56),int(height/2-56))
        bottomRight=(int(width/2+56),int(height/2+56))
        cv2.rectangle(grey,upperLeft,bottomRight,(0,255,0),2)

        roi=grey[upperLeft[1]:bottomRight[1],upperLeft[0]:bottomRight[0]]

        img_pil=Image.fromarray(roi)

        image_bw=img_pil.convert("L")
        image_bw_resized=image_bw.resize((28,28),Image.ANTIALIAS)

        image_bw_resized_inverted=PIL.ImageOps.invert(image_bw_resized)
        pixelFilter=20
        
        min_pixel=np.percentile(image_bw_resized,pixelFilter)

        image_bw_resized_inverted_scaled=np.clip(image_bw_resized_inverted-min_pixel,0,255)
        max_pixel=np.max(image_bw_resized_inverted)

        image_bw_resized_inverted_scaled=np.asarray(image_bw_resized_inverted_scaled)/max_pixel
        testsample=np.array(image_bw_resized_inverted_scaled).reshape(1,784)
        testPrediction=clasifier.predict(testsample)
        print("Predicted class is :",testPrediction)  

        cv2.imshow("frame",grey)
        if cv2.waitKey(1) & 0xFF==ord("q"):
            break
        
    except Exception as e:
        pass      

    
cap.release()
cv2.destroyAllWindows()