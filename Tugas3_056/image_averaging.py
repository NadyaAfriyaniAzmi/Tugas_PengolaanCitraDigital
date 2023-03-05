import cv2
from skimage.util import random_noise
from matplotlib import pyplot as plt

# Load citra yang akan digunakan
img = cv2.imread('image1.jpg')
ori_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Menambahkan gaussian noise ke citra original.
noise_img = random_noise(ori_img, mode='gaussian')

# Menampilkan citra dengan noise
plt.subplot(121), plt.imshow(ori_img), plt.title('Original Image')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(noise_img), plt.title('Image Averaging')
plt.xticks([]), plt.yticks([])
plt.show()