from keras.models import load_model
from keras.preprocessing import image
import numpy as np

model = load_model('model.h5')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


test_image = image.load_img('happy1.jpg', color_mode = "grayscale", target_size = (100, 100)) 
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

result = model.predict(test_image)
print(result)