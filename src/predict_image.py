import tensorflow as tf
import numpy as np
import cv2

# Load model
model = tf.keras.models.load_model("best_efficientnet.h5")

# Class names (must match dataset folders)
class_names = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

# Image path
image_path = "brain_tumor_dataset/Testing/glioma/Te-gl_109.jpg"

# Read image
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Could not load image: {image_path}")

# Check image
if image is None:
    print("Image not found!")
    exit()

# Convert BGR to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Resize
image = cv2.resize(image, (224, 224))

# Normalize
image = image.astype("float32") / 255.0

# Add batch dimension
image = np.expand_dims(image, axis=0)

# Predict
prediction = model.predict(image)

predicted_class = np.argmax(prediction)

confidence = np.max(prediction) * 100

print("\nPrediction :", class_names[predicted_class])
print("Confidence : {:.2f}%".format(confidence))