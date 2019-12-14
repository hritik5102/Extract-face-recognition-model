### Extract-face-recognition-model

### Import OpenCV2 for image processing , Import os for file path 
import cv2, os

### Import numpy for matrix calculation
import numpy as np

### Import Python Image Library (PIL)  
from PIL import Image

### Create Local Binary Patterns Histograms for face recognization 
 recognizer = cv2.face.createLBPHFaceRecognizer() 

### Using prebuilt frontal face training model, for face detection 
face_cascade = cv2.CascadeClassifier('cascade/data/haarcascade_frontalface_alt2.xml')

### Create method to get the images and label data Get the faces and IDs 

### Train the model using the faces and IDs Save the model into trainer.yml 

recognizer.save("trained.yml")
