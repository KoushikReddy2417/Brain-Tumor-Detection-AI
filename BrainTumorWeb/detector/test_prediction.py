import os
from ml.predict import predict_image

# Folder containing testing images
image_path = "../brain_tumor_dataset/Testing/meningioma/Te-me_139.jpg"

# Automatically select the first image
image_name = sorted(os.listdir(folder))[0]

image_path = os.path.join(folder, image_name)

print("Testing:", image_path)

prediction, confidence = predict_image(image_path)

print("Prediction :", prediction)
print("Confidence : {:.2f}%".format(confidence))