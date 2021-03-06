import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage import img_as_float, img_as_ubyte, io


#load image
img = cv2.imread('original.tif')
img = img / 255
#apply non-local means denoise
sigma_est = np.mean(estimate_sigma(img, multichannel = True))
denoise = denoise_nl_means(
    img, 
    h = 1.15 * sigma_est, 
    fast_mode = True, 
    patch_size = 5,
    patch_distance = 3, 
    multichannel = True )
denoise_ubyte = img_as_ubyte(denoise)
#cv2.namedWindow('image', cv2.WINDOW_NORMAL)
#cv2.resizeWindow('image', 600,600)
cv2.imshow('image', denoise_ubyte)
cv2.waitKey(0)
cv2.imwrite('image_denoised.jpg', denoise_ubyte)