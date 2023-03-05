import cv2
import numpy as np
from matplotlib import pyplot as plt

# membaca citra
image = cv2.imread('image2.jpg', 0)

# membuat histogram
hist, bins = np.histogram(image.flatten(), 256, [0, 256])

# melakukan ekualisasi histogram
equa_img = cv2.equalizeHist(image)

#menampilkan citra dan grafik dari citra original dan histogram equalization
plt.subplot(221), plt.imshow(image, cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(equa_img, cmap='gray')
plt.title('Equalized'), plt.xticks([]), plt.yticks([])
plt.subplot(223)
plt.hist(image.flatten(),256,[0,256], color='r')
plt.xlim([0,256])
plt.legend('histogram', loc = 'upper left')
plt.subplot(224)
plt.hist(equa_img.flatten(),256,[0,256], color='r')
plt.xlim([0,256])
plt.legend('histogram', loc= 'upper left')
plt.show()

