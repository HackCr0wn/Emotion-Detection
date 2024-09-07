import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch

# Load in color image for face detection
image = cv2.imread('D:\Downloads\image1.jpeg')

# Convert the image to RGB colorspace
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR to RGB

# Make a copy of the original image to draw face detections on
image_copy = np.copy(image)

# Convert the image to gray 
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Load the pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Detect faces in the image using pre-trained face detector
faces = face_cascade.detectMultiScale(gray_image, 1.25, 4)

# Print number of faces found
print('Number of faces detected:', len(faces))

#Load pre-trained model for Emotion Detection
emotion_model = torch.load('D:\CODE\Python\Emotion Detection\Model')

#List of Emotions
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Suprise', 'Neutral']

# Get the bounding box for each detected face
face_crop = []
for f in faces:
    x, y, w, h = [ v for v in f ]
    cv2.rectangle(image_copy, (x,y), (x+w, y+h), (255,0,0), 3)
    # Define the region of interest in the image  
    face_crop.append(gray_image[y:y+h, x:x+w])

for i,face in enumerate(face_crop):
    #Resizing size of face crop
    face = cv2.resize(face, (48,48))
    face = np.expand_dims(face, axis=0)
    face = np.expand_dims(face, axis=-1)

    #predict emotion of the face
    emotion_prediction = emotion_model.predict(face)
    emotion_label = emotion_labels[np.argmax(emotion_prediction)]

    #display
    cv2.imshow('face',face)
    cv2.waitKey(0)
    print('Faces', i+1, ':', emotion_label)

# Display the image with the bounding boxes
fig, axs = plt.subplots(1, len(face_crop), figsize=(9, 9))
for i, face_crop in enumerate(face_crop):
    axs[i].imshow(face_crop, cmap='gray')
    axs[i].set_title("Face Crop {}".format(i+1))
    axs[i].set_xticks([])
    axs[i].set_yticks([])

# Display the face crops
fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111)
ax.imshow(image_copy)
ax.set_title("Face Detection")
ax.set_xticks([])
ax.set_yticks([])

plt.show()



