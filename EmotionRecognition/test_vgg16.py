from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import cv2

model = load_model('modelofinal.h5')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #Define emotions


faceDet = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faceDet_two = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
faceDet_three = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
faceDet_four = cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")

frame = cv2.imread('imagens/disgust2.jpeg') #Open image
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

#Detect face using 4 different classifiers
face = faceDet.detectMultiScale(gray, 
scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), 
flags=cv2.CASCADE_SCALE_IMAGE)
face_two = faceDet_two.detectMultiScale(gray, 
scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), 
flags=cv2.CASCADE_SCALE_IMAGE)
face_three = faceDet_three.detectMultiScale(gray, 
scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), 
flags=cv2.CASCADE_SCALE_IMAGE)
face_four = faceDet_four.detectMultiScale(gray, 
scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), 
flags=cv2.CASCADE_SCALE_IMAGE)

#Go over detected faces, stop at first detected face, return empty if no face.
if len(face) == 1:
    facefeatures = face
elif len(face_two) == 1:
    facefeatures = face_two
elif len(face_three) == 1:
    facefeatures = face_three
elif len(face_four) == 1:
    facefeatures = face_four
else:
    facefeatures = ""

result = []
out = []

#Cut and save face
for (x, y, w, h) in facefeatures: #get coordinates and size of rectangle containing face
    gray = gray[y:y+h, x:x+w] #Cut the frame to size
    try:
        out = cv2.resize(gray, (150, 150)) #Resize face so all images have same size
    except:
        pass #If error, pass file

#test_image = image.load_img(out, color_mode = "grayscale", target_size = (150, 150))

cv2.imshow('image',out)
cv2.imwrite('output.jpg', out)

test_image = image.img_to_array(out)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)

for j in range(len(result)):
    #print(f"{emotions[j]}: {result[j]*100}")
    print ("Neutral: %", str(round(result[0][0]/1.0 * 100, 4)))
    print ("Anger: %", str(round(result[0][1]/1.0 * 100, 4)))
    print ("Contempt: %", str(round(result[0][2]/1.0 * 100, 4)))
    print ("Disgust: %", str(round(result[0][3]/1.0 * 100, 4)))
    print ("Fear: %", str(round(result[0][4]/1.0 * 100, 4)))
    print ("Happy: %", str(round(result[0][5]/1.0 * 100, 4)))	
    print ("Sadness: %", str(round(result[0][6]/1.0 * 100, 4)))		
    print ("Surprise: %", str(round(result[0][7]/1.0 * 100, 4)))		
    print ("----------------------"	)	
    max_index = np.argmax(result[0])
    emotion = emotions[max_index]
    print (f"Prediction: {emotion}")