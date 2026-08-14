import os
import cv2
import numpy as np

images = []
labels = []

dataset_path = "brain_tumor_dataset/Training"

classes = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

for label, class_name in enumerate(classes):

    folder = os.path.join(dataset_path, class_name)

    files = sorted(os.listdir(folder))

    print(f"\nLoading {class_name}...")

    count = 0

    for file in files:

        image_path = os.path.join(folder, file)

        image = cv2.imread(image_path)

        image = cv2.resize(image, (224,224))

        image = image / 255.0

        images.append(image)

        labels.append(label)

        count += 1

    print("Images Loaded:", count)

print("\nDataset Loading Complete!")

print("Total Images:", len(images))
print("Total Labels:", len(labels))