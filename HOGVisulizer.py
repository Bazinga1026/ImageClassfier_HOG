from skimage.feature import hog
import cv2
import matplotlib.pyplot as plt

img_path = r"Screenshot_98.png"


image = cv2.imread(str(img_path))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_big = cv2.resize(image, (500 , 500))
image_small = cv2.resize(image, (128, 128))

feautresB, imgB  = hog(
        image_big,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        channel_axis=2,
        visualize=True
    )

feautresS, imgS  = hog(
        image_small,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        channel_axis=2,
        visualize=True
    )


fig, axes = plt.subplots(1, 3)

axes[0].imshow(image)
axes[0].axis("off")

axes[1].imshow(imgB)
axes[1].axis("off")
axes[2].imshow(imgS)
axes[2].axis("off")

plt.show()
print(feautresB.shape)
print(feautresS.shape)

