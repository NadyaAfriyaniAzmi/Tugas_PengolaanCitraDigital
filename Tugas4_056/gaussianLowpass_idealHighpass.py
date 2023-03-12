import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv2.imread('lena.jpg', 0)

# Fourier transform
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

# Gaussian lowpass filter
sigma = 30
rows, cols = img.shape
crow, ccol = int(rows/2), int(cols/2)
gaussian = np.exp(-np.power(np.arange(-crow,rows-crow)[:,np.newaxis], 2) / (2 * np.power(sigma, 2))) * np.exp(-np.power(np.arange(-ccol,cols-ccol), 2) / (2 * np.power(sigma, 2)))
fshift_gaussian = fshift * gaussian
f_gaussian = np.fft.ifftshift(fshift_gaussian)
img_gaussian = np.fft.ifft2(f_gaussian).real

# Ideal highpass filter
cutoff_freq = 30
rows, cols = img.shape
crow, ccol = int(rows/2), int(cols/2)
ideal_highpass = np.zeros((rows, cols))
ideal_highpass[crow-cutoff_freq:crow+cutoff_freq, ccol-cutoff_freq:ccol+cutoff_freq] = 1
fshift_ideal_highpass = fshift * ideal_highpass
f_ideal_highpass = np.fft.ifftshift(fshift_ideal_highpass)
img_ideal_highpass = np.fft.ifft2(f_ideal_highpass).real

# Display results
fig, axs = plt.subplots(1, 3, figsize=(10, 5))
axs[0].imshow(img, cmap='gray')
axs[0].set_title('Original Image')
axs[1].imshow(np.uint8(img_gaussian), cmap='gray')
axs[1].set_title('Gaussian Lowpass Filter')
axs[2].imshow(np.uint8(img_ideal_highpass), cmap='gray')
axs[2].set_title('Ideal Highpass Filter')
plt.show()