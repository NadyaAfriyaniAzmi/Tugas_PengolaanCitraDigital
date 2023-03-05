import cv2
import numpy as np

img = cv2.imread("F55121056_Nadya Afriyani azmi_B.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("Gambar Lenna Original", img)
cv2.imshow("Gambar Lenna Grayscale", gray)

cv2.waitKey(0)
cv2.destroyAllWindows()
