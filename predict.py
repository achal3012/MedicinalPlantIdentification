import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

model = tf.keras.models.load_model("model/medicinal_plant_mobilenetv2.keras")

class_names = ["Aloevera", "Neem", "Tulsi"]

img = load_img("static/test.jpg", target_size=(224, 224))

img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

prediction = model.predict(img_array)

predicted_class = class_names[np.argmax(prediction)]
confidence = np.max(prediction)

print("Predicted plant:", predicted_class)
print("Confidence:", confidence)
