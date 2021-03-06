import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage import exposure
from skimage.segmentation import random_walker
from scipy import ndimage as nd
import itertools

#load denosied image
image = cv2.imread('image_denoised.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 600,600)
cv2.imshow('image', image)

# #extend the image histogram by equalising
# eq_img = exposure.equalize_adapthist(image)

# #make markers
# markers = np.zeros(image.shape, dtype=np.uint)
# markers[(eq_img > 0.1) & (eq_img < 0.4)] = 1
# markers[(eq_img > 0.4) & (eq_img < 0.8)] = 2
# markers[(eq_img > 0.8) & (eq_img < 1  )] = 3
# plt.figure()
# plt.imshow(markers)
# plt.show()
# # Run random walker algorithm using sklearn package
# labels = random_walker(eq_img, markers, beta=10, mode='bf')

# segm1 = (labels == 1)
# segm2 = (labels == 2)
# segm3 = (labels == 3)
# all_segments = np.zeros((eq_img.shape[0], eq_img.shape[1], 3))
# all_segments[segm1] = (1,0,0)
# all_segments[segm2] = (0,1,0)
# all_segments[segm3] = (0,0,1)

# segm1_closed = nd.binary_closing(segm1, np.ones((3,3)))
# segm2_closed = nd.binary_closing(segm2, np.ones((3,3)))
# segm3_closed = nd.binary_closing(segm3, np.ones((3,3)))

# all_segments_cleaned = np.zeros((eq_img.shape[0], eq_img.shape[1], 3)) 

# all_segments_cleaned[segm1_closed] = (1,0,0)
# all_segments_cleaned[segm2_closed] = (0,1,0)
# all_segments_cleaned[segm3_closed] = (0,0,1)

# plt.imshow(all_segments_cleaned) 
# plt.show()


#define a function for random_walker algorithm
def randwalk_seg(x):
    #extend the image histogram by equalising
    eq_img = exposure.equalize_adapthist(image)
    #initialise segmentations size
    segdic = {}
    
    all_segments = np.zeros((image.shape[0],image.shape[1] ,3))
    markers = np.zeros(image.shape, dtype=np.uint)
    #initialise colors with combination of R,G,B
    colors = list(itertools.product([0, 1], repeat=3))
    for i in range(x):
        markers[(eq_img >= 1/x*i) & (eq_img < 1/x*i + 1/x)] = i + 1
    labels = random_walker(eq_img, markers, beta=10, mode='bf')
    for i in range(x):
        segdic['seg{}'.format(i)] = (labels == i+1)
        all_segments[segdic['seg{}'.format(i)]] = colors[i+1]
    plt.figure()
    plt.imshow(all_segments)
    plt.axis('off')
    plt.savefig("random_walker.png",bbox_inches='tight',pad_inches=0.0)
    plt.show()
    
    

randwalk_seg(4)

