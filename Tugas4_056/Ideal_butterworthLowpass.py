import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Load image
img = cv2.imread('lena.jpg', 0)

# Fourier transform
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)

# Ideal lowpass filter
rows, cols = img.shape
crow, ccol = int(rows/2), int(cols/2)
cutoff_freq = 50
lowpass = np.zeros((rows, cols))
lowpass[crow-cutoff_freq:crow+cutoff_freq, ccol-cutoff_freq:ccol+cutoff_freq] = 1
fshift_low = fshift * lowpass
f_low = np.fft.ifftshift(fshift_low)
img_low = np.fft.ifft2(f_low).real

# Butterworth lowpass filter
order = 4
cutoff_freq = 50
butterworth = 1 / (1 + (np.sqrt(2)-1)*np.power(np.sqrt(np.power(np.arange(-crow,rows-crow),2)[:,np.newaxis] + np.power(np.arange(-ccol,cols-ccol),2)[np.newaxis,:]), 2*order)/(cutoff_freq*2))
fshift_butterworth = fshift * butterworth
f_butterworth = np.fft.ifftshift(fshift_butterworth)
img_butterworth = np.fft.ifft2(f_butterworth).real


# Display results
cv2.imshow('Original Image', img)
cv2.imshow('Ideal Lowpass Filter', np.uint8(img_low))
cv2.imshow('Butterworth Lowpass Filter', np.uint8(img_butterworth))
cv2.waitKey(0)
cv2.destroyAllWindows()