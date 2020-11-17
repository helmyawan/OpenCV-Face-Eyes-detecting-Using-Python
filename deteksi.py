import cv2
import argparse
import numpy as np  

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml'); #Definisi Variabel Face detek
eyesDetect = cv2.CascadeClassifier('haarcascade_eye.xml'); #Definisi variabel eyes detek
camera = cv2.VideoCapture(0);

while (True):
    ret,img =camera.read(); #Berfungsi sebagai camera agar bisa menangkap object yang di tangkap
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #Untuk mengganti frame pada webcam
    faces = faceDetect.detectMultiScale(gray,1.3,5); 
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y), (x+w,y+h),(0,255,0),2) #Membuat kotak dan membuat warna hijau pada object yang di deteksi
        cv2.imshow("Face",img);
        if(cv2.waitKey(1) ==ord('q')):
            ret,img =camera.read(); #Berfungsi sebagai camera agar bisa menangkap object yang di tangkap
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #Untuk mengganti frame pada webcam
    faces = eyesDetect.detectMultiScale(gray,1.3,5);
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y), (x+w,y+h),(0,255,0),2) #Membuat kotak dan membuat warna hijau pada object yang di deteksi
        cv2.imshow("Face",img);
        if(cv2.waitKey(1) ==ord('q')):
            break;
                                      
camera.release()
cv2.destroyAllwindows()
