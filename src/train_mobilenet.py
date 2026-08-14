import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import pickle

# ===========================================
# Configuration
# ===========================================
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 15

# ===========================================
# Load Dataset
# ===========================================
train_dataset = tf.keras.utils.image_dataset_from_directory(
    "brain_tumor_dataset/Training",
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    "brain_tumor_dataset/Training",
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_dataset.class_names

print("\nClasses:", class_names)

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(AUTOTUNE)
validation_dataset = validation_dataset.prefetch(AUTOTUNE)

# ===========================================
# Data Augmentation
# ===========================================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.10),
    layers.RandomZoom(0.10),
])

# ===========================================
# Load MobileNetV2
# ===========================================
base_model = MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=(224,224,3)
)

# Fine-tune last 30 layers
base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

# ===========================================
# Functional API Model
# ===========================================
inputs = tf.keras.Input(shape=(224,224,3))

x = data_augmentation(inputs)

x = preprocess_input(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.30)(x)

x = layers.Dense(128, activation="relu")(x)

x = layers.Dropout(0.30)(x)

outputs = layers.Dense(4, activation="softmax")(x)

model = Model(inputs, outputs)

# ===========================================
# Compile
# ===========================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n================ MODEL SUMMARY ================\n")
model.summary()

# ===========================================
# Callbacks
# ===========================================
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath="best_mobilenet.h5",
    monitor="val_loss",
    save_best_only=True,
    mode="min"
)

# ===========================================
# Train
# ===========================================
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint]
)

# ===========================================
# Save History
# ===========================================
with open("mobilenet_history.pkl", "wb") as f:
    pickle.dump(history.history, f)

# ===========================================
# Save Final Model
# ===========================================
model.save("mobilenet_final.h5")

print("\n======================================")
print("Training Completed Successfully!")
print("======================================")