import tensorflow as tf
from tensorflow.keras import layers, models

train_dataset = tf.keras.utils.image_dataset_from_directory(
    "brain_tumor_dataset/Training",
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=(224,224),
    batch_size=16
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    "brain_tumor_dataset/Training",
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(224,224),
    batch_size=16
)

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.shuffle(500).prefetch(AUTOTUNE)
validation_dataset = validation_dataset.prefetch(AUTOTUNE)

model = models.Sequential([

    layers.Conv2D(32,(3,3),activation='relu',input_shape=(224,224,3)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(128,(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(128,activation='relu'),

    layers.Dropout(0.5),

    layers.Dense(4,activation='softmax')

])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=3
)
import pickle

with open("training_history.pkl", "wb") as file:
    pickle.dump(history.history, file)
model.save("brain_tumor_model.keras")

print("Model Saved Successfully!")