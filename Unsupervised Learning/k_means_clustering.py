import pandas as pd
import cv2
import numpy as np

#load denosied image
image = cv2.imread('image_denoised.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 600,600)
cv2.imshow('image', image)
# cv2.waitKey(0)

#reshape to n *1 * 1 array
image2 = image.reshape((-1, 1))
image2 = np.float32(image2)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.1)

#CLUSTERS 
k = 3
attempts = 15
ret, label, center = cv2.kmeans(image2, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
center = np.uint8(center)

res = center[label.flatten()]
res2 = res.reshape((image.shape))

cv2.namedWindow('res2', cv2.WINDOW_NORMAL)
cv2.resizeWindow('res2', 600,600)
cv2.imshow('res2', res2)
cv2.waitKey(0)
result=cv2.imwrite('k_means_clustering.png', res2)