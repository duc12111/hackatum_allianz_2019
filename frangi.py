import cv2
from skimage.filters import frangi
import matplotlib.pyplot as plt

img = cv2.imread('data/data_scaled/201312002_LIMI_008257.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

fig, ax = plt.subplots()

ax.imshow(frangi(img), cmap=plt.cm.gray_r)
ax.set_title('Frangi filter result')

ax.axis('off')

plt.tight_layout()
plt.show()