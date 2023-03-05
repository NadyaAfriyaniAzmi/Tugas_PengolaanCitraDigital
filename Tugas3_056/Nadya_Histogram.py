import cv2
import numpy as np
from matplotlib import pyplot as plt

# membaca Gambar
image = cv2.imread('image2.jpg', 0)

# membuat histogram
hist, bins = np.histogram(image.flatten(), 256, [0, 256])

# mencari nilai intensitas piksel yang berfrekuensi maximal
max_intensity = np.argmax(hist)

# membuat lookup table untuk menggeser nilai intensitas piksel
lut = np.zeros((256,), dtype=np.uint8)
for i in range(256):
    lut[i] = np.uint8(np.clip((i - max_intensity) * 1.5 + max_intensity, 0, 255))

# mengaplikasikan lookup table ke citra
rst_img = cv2.LUT(image, lut)

#menampilkan citra dan grafik dari citra original dan result
plt.subplot(221), plt.imshow(image, cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(rst_img, cmap='gray')
plt.title('Result'), plt.xticks([]), plt.yticks([])
plt.subplot(223)
plt.hist(image.flatten(),256,[0,256], color='r')
plt.xlim([0,256])
plt.legend('histogram', loc = 'upper left')
plt.subplot(224)
plt.hist(rst_img.flatten(),256,[0,256], color='r')
plt.xlim([0,256])
plt.legend('histogram', loc= 'upper left')
plt.show()
