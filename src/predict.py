import tensorflow as tf
import numpy as np

MODEL_PATH = "models/mobilenetv2.keras"

model = tf.keras.models.load_model(MODEL_PATH)

class_names = ["drive","legglance-flick","pullshot","sweep"]


def predict_image(img):

    img = tf.image.resize(img, (256,256))
    img = img / 255.0
    img_array = tf.expand_dims(img, 0)

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))

    return predicted_class, confidence