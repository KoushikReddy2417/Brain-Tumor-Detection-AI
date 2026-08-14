import os
import cv2

folder = "../brain_tumor_dataset/Training/glioma"

files = sorted(os.listdir(folder))

images = []

for file in files:

    image_path = os.path.join(folder, file)

    image = cv2.imread(image_path)

    image = cv2.resize(image, (224, 224))

    image = image / 255.0

    images.append(image)

print("Total Images Loaded:", len(images))

print("Shape of First Image:", images[0].shape)