import tensorflow as tf


model = tf.keras.models.load_model("real_or_ai_classifier.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("real_or_ai_classifier.tflite", "wb") as f:
    f.write(tflite_model)
