from keras.models import load_model
from keras.preprocessing import image
import numpy as np

emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #Define emotions

model = load_model('model_final.h5')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

test_image = image.load_img('happy_2.jpeg', color_mode = "rgb", target_size = (150, 150)) 
test_image = image.img_to_array(test_image)
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