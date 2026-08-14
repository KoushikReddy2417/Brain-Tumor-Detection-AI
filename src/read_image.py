import cv2

image = cv2.imread("../brain_tumor_dataset/Training/glioma/Tr-gl_1000.jpg")
image = cv2.resize(image, (224, 224))

normalized_image = image / 255.0

print("Top Left Pixel:")
print(image[0,0])

print()

print("Center Pixel:")
print(image[112,112])

print()

print("Normalized Center Pixel:")
print(normalized_image[112,112])
