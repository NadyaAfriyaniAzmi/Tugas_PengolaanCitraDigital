import numpy as np
import cv2

def median_filter(image, kernel_size):
    height, width = image.shape
    pad_size = kernel_size // 2
    padded_image = np.pad(image, pad_size, mode='reflect')
    filtered_image = np.zeros((height, width), dtype=np.uint8)
    for i in range(pad_size, height+pad_size):
        for j in range(pad_size, width+pad_size):
            filtered_image[i-pad_size, j-pad_size] = np.median(padded_image[i-pad_size:i+pad_size+1, j-pad_size:j+pad_size+1])
    return filtered_image

def min_filter(image, kernel_size):
    height, width = image.shape
    pad_size = kernel_size // 2
    padded_image = np.pad(image, pad_size, mode='reflect')
    filtered_image = np.zeros((height, width), dtype=np.uint8)
    for i in range(pad_size, height+pad_size):
        for j in range(pad_size, width+pad_size):
            filtered_image[i-pad_size, j-pad_size] = np.min(padded_image[i-pad_size:i+pad_size+1, j-pad_size:j+pad_size+1])
    return filtered_image

def max_filter(image, kernel_size):
    height, width = image.shape
    pad_size = kernel_size // 2
    padded_image = np.pad(image, pad_size, mode='reflect')
    filtered_image = np.zeros((height, width), dtype=np.uint8)
    for i in range(pad_size, height+pad_size):
        for j in range(pad_size, width+pad_size):
            filtered_image[i-pad_size, j-pad_size] = np.max(padded_image[i-pad_size:i+pad_size+1, j-pad_size:j+pad_size+1])
    return filtered_image

# Read the image in grayscale
image = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)

# Apply median filter with kernel size 3x3
median_filtered = median_filter(image, 3)

# Apply min filter with kernel size 5x5
min_filtered = min_filter(image, 5)

# Apply max filter with kernel size 7x7
max_filtered = max_filter(image, 7)

# Display the original image and the filtered images
cv2.imshow('Original Image', image)
cv2.imshow('Median Filtered Image', median_filtered)
cv2.imshow('Min Filtered Image', min_filtered)
cv2.imshow('Max Filtered Image', max_filtered)

# Wait for a key press and then exit
cv2.waitKey(0)
cv2.destroyAllWindows()
