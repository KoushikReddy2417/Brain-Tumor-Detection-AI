import tensorflow as tf

train_dataset = tf.keras.utils.image_dataset_from_directory(
    "brain_tumor_dataset/Training",
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(224,224),
    batch_size=32
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    "brain_tumor_dataset/Training",
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(224,224),
    batch_size=32
)

# Improve performance
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.cache().shuffle(1000).prefetch(AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(AUTOTUNE)

print("Training Dataset Ready")
print(train_dataset)

print("\nValidation Dataset Ready")
print(validation_dataset)