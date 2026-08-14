import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Load trained model
model = tf.keras.models.load_model("brain_tumor_model.h5")

# Load test dataset
test_dataset = tf.keras.utils.image_dataset_from_directory(
    "brain_tumor_dataset/Testing",
    image_size=(224, 224),
    batch_size=32,
    shuffle=False
)

# Get class names
class_names = test_dataset.class_names
print("Classes:", class_names)

# Get true labels
y_true = np.concatenate([labels.numpy() for images, labels in test_dataset])

# Predict
predictions = model.predict(test_dataset)
y_pred = np.argmax(predictions, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=300)
print("Confusion matrix saved to results/confusion_matrix.png")

# Classification Report
print("\nClassification Report\n")
print(classification_report(
    y_true,
    y_pred,
    target_names=class_names
))