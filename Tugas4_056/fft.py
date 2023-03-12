import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load gambar
img = cv2.imread('lena.jpg', 0)

# Hitung FFT pada gambar
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

# Tampilkan gambar dan spektrum magnitudo FFT
plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Gambar Asli'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Spektrum Magnitudo FFT'), plt.xticks([]), plt.yticks([])
plt.show()