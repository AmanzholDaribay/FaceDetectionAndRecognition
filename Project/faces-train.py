import cv2
import os
import numpy as np
from PIL import Image
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) #it gives the just a path of this file faces-train.y
image_dir = os.path.join(BASE_DIR, "images")

face_cascade1 = cv2.CascadeClassifier('data/haarcascade_frontalface_alt2.xml') #for detection
recognizer = cv2.face.LBPHFaceRecognizer_create() #for comparison of detected with in database

current_id = 0
label_ids = {} #create a dictionary and to extract the data from it, you need
#the syntax is: mydict[key] = "value"

y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir): 
	for file in files:
		if file.endswith("png") or file.endswith("jpg"):
			path = os.path.join(root, file) #it shows the place of image
			#the path will become kind of imgs\file, without join it would be just file
			label = os.path.basename(root).replace(" ", "_").lower() #the base name is for imgs\file is imgs, and 
			#it takes it and replaces " " to "_"
		

			#here we are just creating an id for each label
			#such as label is Amanzhol Daribay and id is 0.
			if label in label_ids: 
				pass
			else:
				label_ids[label] = current_id
				current_id += 1

				

			id_ = label_ids[label] #if it is in the label we take it from the label_ids
			#else we create the id by current_id
			#and equate it to the id for the future

			pil_image = Image.open(path).convert("L") #conversion the image into grayscale
			#pil_image is the pixels of gray so only one matrix
			image_array = np.array(pil_image, "uint8") #converted these grayscale
			#every pixel value to some sort of numpy array (into numbers)
			

			#Before this point we were finding the images and converting them into
			#numpy array

			#Next we are extracting the faces from these arrays


			faces1 = face_cascade1.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)
			#what we do is actually the same thing as detecting the face and returning 
			#x,y,w,h
			#since detectMutliScale worked finding in pixels, which are also numbers
			#it should work for a image_array, which is also a bunch of numbers
			#but in the form of numpy arrays
			#This was made by the author of a video

			for (x,y,w,h) in faces1:
				roi = image_array[y:y+h, x:x+h] #we are cutting the face place and 
				#call it roi = region of interes
				x_train.append(roi) #we put this roi into the training dataset
				y_labels.append(id_) #before we had label_ids with label and id
				#this list is only for ids
				#also if you have a huge number of photos of one person of one label
				#all of them are realted to one id

				#BY DOING ALL THEES STUFF WE WANT TO SAY TO A MACHINE THAT ALL 
				#SPECIFIC x_train, which are pixels in numpy
				#ARE RELATED TO A SPECIFIC label in an y_labels

#print(x_train)
#print(y_labels)


#Pickling - is the process whereby a Python 
#object hierarchy is converted into a byte stream

# pyhton list for pickling is label_ids

#file name to store serialized data
file_Name = "labels.pickle"

#open the file for writing
fileObject = open(file_Name, 'wb')

#this writes the label_ids into
#labels.pickle 
pickle.dump(label_ids, fileObject)


#WHAT IS PICKLE?????????????????????


recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainner.yml")




