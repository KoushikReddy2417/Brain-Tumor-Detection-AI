import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

model = tf.keras.models.load_model("best_efficientnet.h5")

test_dataset = tf.keras.utils.image_dataset_from_directory(
    "brain_tumor_dataset/Testing",
    image_size=(224,224),
    batch_size=16,
    shuffle=False
)

# Apply the same preprocessing used during training
test_dataset = test_dataset.map(
    lambda x, y: (preprocess_input(tf.cast(x, tf.float32)), y)
)

loss, accuracy = model.evaluate(test_dataset)

print("\nTest Loss:", loss)
print("Test Accuracy:", accuracy)