# -*- coding: utf-8 -*-
"""

@author: wile3coyote
"""
from keras.models import load_model
import mtcnn as mt
import numpy as np
from PIL import Image
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
import os



model = load_model('model/facenet_keras.h5')


detector = mt.mtcnn.MTCNN()

# function for face detection with mtcnn
def extract_face(filename, required_size=(160, 160)):
	image = Image.open(filename)
	image = image.convert('RGB')
	pixels = asarray(image)
	detector = MTCNN()
	results = detector.detect_faces(pixels)
	x1, y1, width, height = results[0]['box']
	x1, y1 = abs(x1), abs(y1)
	x2, y2 = x1 + width, y1 + height
	face = pixels[y1:y2, x1:x2]
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array


#function for loading faces from one folder
def load_faces(directory):
	faces=list()
	for filename in os.listdir(directory):
		path=directory+filename
		face=extract_face(path)
		faces.append(face)
	return faces

#function for loading faces and labels by folder
def load_dataset(directory):
	X,y=list(),list()
	for subdir in os.listdir(directory):
		path=directory+subdir+'/'
		if not os.path.isdir(path):
			continue
		faces=load_faces(path)
		labels=[subdir for _ in range(len(faces))]
		print('>loaded %d examples for class: %s' % (len(faces), subdir))
		X.extend(faces)
		y.extend(labels)
	return np.asarray(X),np.asarray(y)

trainX,trainy=load_dataset('data/')

np.savez_compressed('embeddings_dataset.npz',trainX,trainy)

print("Created embeddings")
