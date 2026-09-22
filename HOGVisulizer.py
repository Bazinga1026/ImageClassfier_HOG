from skimage.feature import hog
import cv2
import matplotlib.pyplot as plt

img_path = r"HOG\dataset\caltech-101\faces\image_0001.jpg"


image = cv2.imread(str(img_path))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (128, 128))


feautres, img  = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        channel_axis=2,
        visualize=True
    )


fig, axes = plt.subplots(1, 2)

axes[0].imshow(image)
axes[0].axis("off")

axes[1].imshow(img)
axes[1].axis("off")

plt.show()
print(feautres.shape)

