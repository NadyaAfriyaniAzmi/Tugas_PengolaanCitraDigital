import cv2
#Citra Asli
img = cv2.imread('imglogin.jfif', 0)
#Mengubah Citra
img2 = cv2.absdiff(cv2.GaussianBlur(img, (9, 9), 0), img)

# Tampilan hasil perubahan citra
cv2.imshow('Original Image', img)
cv2.imshow('Difference Image', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()