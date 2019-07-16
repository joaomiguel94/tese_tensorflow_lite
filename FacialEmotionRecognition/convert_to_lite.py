import tensorflow as tf

converter = tf.contrib.lite.TFLiteConverter.from_saved_model("model.h5")
tflite_model = converter.convert()
open("model_lite.tflite", "wb").write(tflite_model)
