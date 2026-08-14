import tensorflow as tf

model = tf.keras.models.load_model("mobilenet_final.h5")

test_dataset = tf.keras.utils.image_dataset_from_directory(
    "brain_tumor_dataset/Testing",
    image_size=(224,224),
    batch_size=16,
    shuffle=False
)

loss, accuracy = model.evaluate(test_dataset)

print("\nTest Loss :", loss)
print("Test Accuracy :", accuracy)