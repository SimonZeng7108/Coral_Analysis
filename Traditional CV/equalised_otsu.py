import cv2
import numpy as np
from matplotlib import pyplot as plt

#load denosied image
image = cv2.imread('image_denoised.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 600,600)
cv2.imshow('image', image)

#Contrast Limited Adaptive Histogram Equalization (CLAHE)
clahe = cv2.createCLAHE(clipLimit = 3, tileGridSize=(8,8))
cl_image = clahe.apply(image)


#otsu segmentation
ret2, thresh2 = cv2.threshold(cl_image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
kernel = np.ones((3,3), np.uint8)
erosion = cv2.erode(thresh2, kernel, iterations = 1)
dilation = cv2.dilate(erosion, kernel, iterations = 1)
#erosion + dilation  are same as cv2.morphologyEX
cv2.namedWindow('OTSU', cv2.WINDOW_NORMAL)
cv2.resizeWindow('OTSU', 600,600)
cv2.imshow('OTSU', dilation)
cv2.waitKey(0)

result=cv2.imwrite('otsu.png', dilation)


# #riddler-calvard
# import numpy as np
# import mahotas
# import cv2
# #load denosied image
# image = cv2.imread('image_denoised.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# # cv2.resizeWindow('image', 600,600)
# # cv2.imshow('image', image)

# T = mahotas.thresholding.rc(image)
# thresh = image.copy()
# thresh[thresh > T] = 255
# thresh[thresh < 255] = 0
# # thresh = cv2.bitwise_not(thresh)
# cv2.namedWindow('Riddler-Calvard', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Riddler-Calvard', 600,600)
# cv2.imshow('Riddler-Calvard', thresh)

# cv2.waitKey(0)