from cv2 import cv2
import numpy as np

camera_port=0
video=cv2.VideoCapture(camera_port, cv2.CAP_DSHOW)


while True:
    ret, img = video.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    facecascade = cv2.CascadeClassifier('D:\\Pranav\\Python\\haarcascade_frontalface_default.xml')
    faces = facecascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,     
        minSize=(20, 20)
    )

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        

    cv2.imshow('video',img)

    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break

video.release()
cv2.destroyAllWindows()