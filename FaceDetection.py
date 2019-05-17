import cv2
import numpy as np
import glob

#Reading the series of images
txtfiles = [] 
for file in glob.glob("*.jpg"):
    txtfiles.append(file)

#Reading and converting the images into greyscale
for ix in txtfiles:
    img = cv2.imread(ix,cv2.IMREAD_COLOR)
    imgtest1 = img.copy()
    imgtest = cv2.cvtColor(imgtest1, cv2.COLOR_BGR2GRAY)
    facecascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = facecascade.detectMultiScale(imgtest, scaleFactor=1.2, minNeighbors=6)

#Recording the number of faces
    print('Number of face found',len(faces))

#Drawing the squares on each faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 2)
        roi_gray = imgtest[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
#This will output the images into a video mode
    cv2.imshow('Face Detection',img)

#Exits the program
    k = cv2.waitKey(30) & 0xff
    if k == 27: 
        break


