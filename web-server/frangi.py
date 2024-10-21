import cv2
from skimage.filters import frangi
import matplotlib.pyplot as plt
from PIL import Image

def apply_frangi(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    
    new_image = frangi(img)
    show_image(new_image)
    return new_image

def show_image(img):
    fig, ax = plt.subplots()
    
    ax.imshow(img, cmap=plt.cm.gray_r)
    ax.set_title('Frangi filter result')

    ax.axis('off')

    plt.tight_layout()
    plt.show()