# -*- coding: utf-8 -*-
"""

@author: punit
"""
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
import cv2
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from PIL import Image
import os
from mtcnn.mtcnn import MTCNN


from keras.models import load_model

data=np.load('embeddings_dataset.npz')
detector = MTCNN()

model = load_model('model/facenet_keras.h5')

def get_embedding(model,face_pixels):
	face_pixels=face_pixels.astype('float32')
	mean, std = face_pixels.mean(),face_pixels.std()
	face_pixels=(face_pixels-mean)/std
	samples=np.expand_dims(face_pixels,axis=0)
	yhat=model.predict(samples)
	return yhat[0]

trainX,trainy=data['arr_0'],data['arr_1']

newTrainX=list()
for face_pixels in trainX:
	embedding=get_embedding(model,face_pixels)
	newTrainX.append(embedding)

newTrainX=np.asarray(newTrainX)

in_encoder=Normalizer('l2')
trainX=in_encoder.transform(newTrainX)
out_encoder=LabelEncoder()
out_encoder.fit(trainy)
trainy=out_encoder.fit_transform(trainy)

model_svc=SVC(kernel='linear')
model_svc.fit(trainX,trainy)
yhat_train=model_svc.predict(trainX)
score_train=accuracy_score(trainy,yhat_train)
print(score_train)

def extract_face_mod(image, required_size=(160, 160)):
	
	img = image
	pixels = np.asarray(image)
	detector = MTCNN()
	results = detector.detect_faces(pixels)

	x1, y1, width, height = results[0]['box']
	x1, y1 = abs(x1), abs(y1)
	x2, y2 = x1 + width, y1 + height
	face = pixels[y1:y2, x1:x2]
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = np.asarray(image)
	return face_array,x1,y1,x2,y2,np.asarray(img)	


cap = cv2.VideoCapture(0)
while True: 
		__, frame = cap.read()
		
		result = detector.detect_faces(frame)
		if result != []:
				for person in result:
					face_pixel,x1,y1,x2,y2,image=extract_face_mod(frame)
					face_emb=get_embedding(model,face_pixel)
					sample=np.expand_dims(face_emb,axis=0)
					yhat=model_svc.predict(sample)
					face_name=out_encoder.inverse_transform(yhat)
					# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
					cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255))
					font = cv2.FONT_HERSHEY_SIMPLEX
					cv2.putText(frame,face_name[0],(x1-10,y1-10), font, 0.5,(0,0,255),1,cv2.LINE_AA)
		cv2.imshow('Image',frame)

		if cv2.waitKey(1) &0xFF == ord('q'):
				break

cap.release()
cv2.destroyAllWindows()
		

