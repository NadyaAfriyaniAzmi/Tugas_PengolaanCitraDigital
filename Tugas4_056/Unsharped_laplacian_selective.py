import cv2
import numpy as np

def unsharp_masking(image, ksize, sigma, alpha, beta):
    # konversi gambar ke float dan lakukan filter Gaussian
    image = image.astype(np.float32)
    blurred = cv2.GaussianBlur(image, ksize, sigma)

    # hitung citra detail
    detail = image - blurred

    # gabung citra detail dengan gambar awal dengan faktor pengali alpha dan beta
    sharpened = cv2.addWeighted(image, alpha, detail, beta, 0)

    # normalisasi gambar
    sharpened = cv2.normalize(sharpened, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    return sharpened

def laplacian_frequency_domain(image):
    # konversi gambar ke float dan hitung dimensi Fourier
    f = np.fft.fft2(image.astype(float))
    fshift = np.fft.fftshift(f)

    # buat filter Laplacian di domain frekuensi
    rows, cols = image.shape
    x = np.arange(cols) - int(cols/2)
    y = np.arange(rows) - int(rows/2)
    xx, yy = np.meshgrid(x, y, sparse=True)
    d = np.sqrt(xx*2 + yy*2)
    H = -4 * np.pi*2 * d*2

    # aplikasikan filter ke spektrum citra Fourier
    filtered = fshift * H

    # konversi kembali ke citra spasial
    f_ishift = np.fft.ifftshift(filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.real(img_back)

    # normalisasi citra
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    return img_back

def selective_filter(image, ksize, threshold):
    # konversi gambar ke float
    image = image.astype(np.float32)

    # lakukan filter median pada gambar
    median = cv2.medianBlur(image, ksize)

    # hitung perbedaan antara gambar asli dan gambar median
    diff = np.abs(image - median)

    # buat mask dengan threshold yang ditentukan
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask[diff > threshold] = 1

    # terapkan mask ke gambar asli
    filtered = cv2.bitwise_and(image, image, mask=mask)

    # normalisasi citra
    filtered = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    return filtered

# Load gambar
image = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)

# Terapkan Unsharp Masking
sharpened = unsharp_masking(image, (5, 5), 1.5, 1.5, -0.5)

# Terapkan Laplacian di domain frekuensi
filtered_laplacian = laplacian_frequency_domain(image)

# Terapkan Selective filter
filtered_selective = selective_filter(image, 5, 10)

# Tampilkan hasil
cv2.imshow('Original Image', image)
cv2.imshow('sharpened Image', sharpened)
cv2.imshow('laplacian Image', filtered_laplacian)
cv2.imshow('selective Image', filtered_selective)
cv2.waitKey(0)
cv2.destroyAllWindows()
