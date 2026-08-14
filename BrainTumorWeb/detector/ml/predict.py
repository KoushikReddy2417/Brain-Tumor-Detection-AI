import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.applications.efficientnet import preprocess_input

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "best_efficientnet.h5"
)

model = tf.keras.models.load_model(MODEL_PATH)

classes = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]


def predict_image(image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (224, 224))

    image = image.astype(np.float32)

    image = preprocess_input(image)

    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image, verbose=0)[0]

    print("\n========== MODEL DEBUG ==========")
    print("Prediction Vector :", prediction)
    print("Glioma      :", prediction[0])
    print("Meningioma  :", prediction[1])
    print("No Tumor    :", prediction[2])
    print("Pituitary   :", prediction[3])
    print("Predicted Index :", np.argmax(prediction))
    print("=================================\n")

    print("\n========== DEBUG ==========")
    print("Prediction Vector:", prediction)
    print("Argmax:", np.argmax(prediction))
    print("===========================\n")

    predicted_class = classes[np.argmax(prediction)]

    confidence = float(np.max(prediction) * 100)

    probabilities = {
        classes[i]: round(float(prediction[i] * 100), 2)
        for i in range(4)
    }

    return predicted_class, confidence, probabilities