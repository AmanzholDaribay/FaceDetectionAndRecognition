import numpy as np
import cv2

face_cascade1 = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml')
face_cascade2 = cv2.CascadeClassifier('data/haarcascade_frontalcatface.xml')
face_cascade3 = cv2.CascadeClassifier('data/haarcascade_frontalcatface_extended.xml')
face_cascade4 = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')
face_cascade5 = cv2.CascadeClassifier('data/haarcascade_profileface.xml')
face_cascade6 = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create() #for comparison of detected with in database
recognizer.read("trainner.yml") #we bring our trainned recognizer
#it means trainner.yml is just tuned theta and etc features which relate input with output

cap = cv2.VideoCapture(0) #turning on camera
cnt=0 #for the images

while True:
  cnt+=1 #this made for numbering outputed images
  ret, frame = cap.read() #read from camera
  #reading for the cap is viewing it as a frames
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  #converting each frame into gray for convenience
  faces1 = face_cascade1.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
  faces2 = face_cascade2.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
  faces3 = face_cascade3.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
  faces4 = face_cascade4.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
  faces5 = face_cascade5.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
  faces6 = face_cascade6.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
  #these just find faces in the gray
  #and give the x,y,w,h and that's set of numbers but not the image of a face


  for (x, y, w, h) in faces6:
    #print (x, y, w, h)
    roi_gray = gray[y:y+h, x:x+w] #this colors this region into gray and cuts outs, naming
    #it as a roi_gray which means region of interest with gray
    #having a gray is very convenient for a classifier

    id_, conf = recognizer.predict(roi_gray) #so recognizer predict to which class or id 
    #the roi_gray is related
    #conf stands for confidence which is something
    #kind of cloziness to the real value of id_
    if conf >= 45 and conf <= 85:
      print(id_)

    img_item = "imgs/my-image"+str(cnt)+".png" #this creates just a name in the folder imgs
    cv2.imwrite(img_item, roi_gray) #this function of cv2 puts the value of roi_gray into
    #img_item

    color = (255, 0, 0)
    stroke = 2 #thikness of a rectangle
    end_x = x+w
    end_y = y+h
    cv2.rectangle(frame, (x, y), (end_x, end_y), color, stroke) #draws a rectangle
    #around the face


  #The rest of the code is in the same manner of for loop above, which cuts, grays
  #and saves the detected faces based on Cascades mentioned at the top
  for (x, y, w, h) in faces1:
    #print (x, y, w, h)
    roi_gray = gray[y:y+h, x:x+w]
    img_item = "imgs/my-image"+str(cnt)+".png"
    cv2.imwrite(img_item, roi_gray)
    
    color = (255, 255, 0)
    stroke = 2
    end_x = x+w
    end_y = y+h
    cv2.rectangle(frame, (x, y), (end_x, end_y), color, stroke)
  
  for (x, y, w, h) in faces2:
    #print (x, y, w, h)
    roi_gray = gray[y:y+h, x:x+w]
    img_item = "imgs/my-image"+str(cnt)+".png"
    cv2.imwrite(img_item, roi_gray)
    
    color = (255, 0, 255)
    stroke = 2
    end_x = x+w
    end_y = y+h
    cv2.rectangle(frame, (x, y), (end_x, end_y), color, stroke)
  
  for (x, y, w, h) in faces3:
    #print (x, y, w, h)
    roi_gray = gray[y:y+h, x:x+w]
    img_item = "imgs/my-image"+str(cnt)+".png"
    cv2.imwrite(img_item, roi_gray)
    
    color = (255, 255, 255)
    stroke = 2
    end_x = x+w
    end_y = y+h
    cv2.rectangle(frame, (x, y), (end_x, end_y), color, stroke)
  
  for (x, y, w, h) in faces4:
    #print (x, y, w, h)
    roi_gray = gray[y:y+h, x:x+w]
    img_item = "imgs/my-image"+str(cnt)+".png"
    cv2.imwrite(img_item, roi_gray)
    
    color = (0, 255, 0)
    stroke = 2
    end_x = x+w
    end_y = y+h
    cv2.rectangle(frame, (x, y), (end_x, end_y), color, stroke)
  
  for (x, y, w, h) in faces5:
    #print (x, y, w, h)
    roi_gray = gray[y:y+h, x:x+w]
    img_item = "imgs/my-image"+str(cnt)+".png"
    cv2.imwrite(img_item, roi_gray) 
    
    color = (0, 0, 255)
    stroke = 2
    end_x = x+w
    end_y = y+h
    cv2.rectangle(frame, (x, y), (end_x, end_y), color, stroke)
  
  cv2.imshow('Hello, Amanzhol', frame) #before cv2 was reading from camera by cv3.read
  #here it is showing it by cv2.imshow
  if (cv2.waitKey(20) & 0xFF == ord('q')): #this is for quitting from "My Frame"
    break #this breaks the while
cap.release()
cv2.destroyAllWindows()
