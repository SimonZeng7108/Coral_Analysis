import cv2
import numpy as np
from sklearn.mixture import GaussianMixture as GMM
import matplotlib.pyplot as plt

#load denosied image
image = cv2.imread('image_denoised.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 600,600)
cv2.imshow('image', image)
# cv2.waitKey(0)

#reshape image
image2 = image.reshape((-1, 1))

#build the model
gmm_model = GMM(n_components=5, covariance_type = 'tied').fit(image2)
gmm_labels = gmm_model.predict(image2)

image3 = gmm_labels.reshape(image.shape[0], image.shape[1])
image3 = np.uint8(image3)
image3[image3 == 0] = 0
image3[image3 == 1] = 255
cv2.namedWindow('converted', cv2.WINDOW_NORMAL)
cv2.resizeWindow('converted', 600,600)
cv2.imshow('converted', image3)
cv2.waitKey(0)

result=cv2.imwrite('GMM.png', image3)