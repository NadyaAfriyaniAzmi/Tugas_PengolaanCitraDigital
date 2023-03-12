import cv2
import numpy as np

def gaussian_highpass_filter(image, ksize, sigma):
    # konversi gambar ke float dan lakukan filter Gaussian
    image = image.astype(np.float32)
    blurred = cv2.GaussianBlur(image, ksize, sigma)

    # lakukan operasi high-pass untuk mendapatkan edge dari gambar
    filtered = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

    # normalisasi gambar
    filtered = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    return filtered

def butterworth_highpass_filter(image, cutoff, order):
    # konversi gambar ke float dan hitung dimensi Fourier
    f = np.fft.fft2(image.astype(float))
    fshift = np.fft.fftshift(f)

    # buat filter Butterworth highpass
    rows, cols = image.shape
    x = np.arange(cols) - int(cols/2)
    y = np.arange(rows) - int(rows/2)
    xx, yy = np.meshgrid(x, y, sparse=True)
    d = np.sqrt(xx*2 + yy*2)
    H = 1 / (1 + (d / cutoff)**(2*order))

    # aplikasikan filter ke spektrum citra Fourier
    filtered = fshift * H

    # konversi kembali ke citra spasial
    f_ishift = np.fft.ifftshift(filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.real(img_back)

    # normalisasi citra
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    return img_back

# Load gambar
image = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)

# Terapkan Gaussian highpass filter
filtered_gaussian = gaussian_highpass_filter(image, (5, 5), 1.5)

# Terapkan Butterworth highpass filter
filtered_butterworth = butterworth_highpass_filter(image, 50, 5)

# Tampilkan hasil
cv2.imshow('Original', image)
cv2.imshow('Gaussian Highpass', filtered_gaussian)
cv2.imshow('Butterworth Highpass', filtered_butterworth)
cv2.waitKey(0)
cv2.destroyAllWindows()