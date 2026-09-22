import cv2
import time
import numpy as np
from skimage.feature import hog
from pathlib import Path

def extract_features(image_path):
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize so every image has the same dimensions
    image = cv2.resize(image, (128, 128))

    # HOG
    hog_features = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        channel_axis=2
    )


    return hog_features



start_time = time.time()
X = []
y = []

dataset_path = Path(r"dataset\caltech-101")

for class_folder in dataset_path.iterdir():
    if not class_folder.is_dir():
        continue

    for image_path in class_folder.iterdir():
        if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        features = extract_features(image_path)

        X.append(features)
        y.append(class_folder.name)



X = np.array(X)
y = np.array(y)
print(y[:10])
print(y[-10:])


print(X.shape)
print(type(y))

print((time.time() - start_time))

np.save("X.npy", X)
np.save("y.npy", y)