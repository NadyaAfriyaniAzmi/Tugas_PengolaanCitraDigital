import cv2
import numpy as np

# Load image
img = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)

# Perform DFT
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)

# Rearrange the quadrants of the DFT to center the low frequency
dft_shift = np.fft.fftshift(dft)

# Compute magnitude spectrum
magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]))

# Display images
cv2.imshow('Image', img)
cv2.imshow('DFT', magnitude_spectrum)
cv2.waitKey(0)
cv2.destroyAllWindows()