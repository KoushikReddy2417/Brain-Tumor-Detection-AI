import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("brain_tumor_model.h5")

# Load test dataset
test_dataset = tf.keras.utils.image_dataset_from_directory(
    "brain_tumor_dataset/Testing",
    image_size=(224, 224),
    batch_size=32,
    shuffle=False
)

# Evaluate model
loss, accuracy = model.evaluate(test_dataset)

print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")