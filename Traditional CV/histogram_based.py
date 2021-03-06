import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage as nd
import itertools

#load denosied image
image = cv2.imread('image_denoised.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 600,600)
cv2.imshow('image', image)
# cv2.waitKey(0)
print(np.shape(image))


#basecase 

# #plot histogram
# plt.figure()
# plt.hist(image.flat, bins = 100, range = (0, 255))

# #initiate segmentation
# segm1 = (image <= 55)
# segm2 = (image > 55) & (image <= 150)
# segm3 = (image > 150) & (image <= 190)
# segm4 = (image > 190)
# all_segments = np.zeros((image.shape[0],image.shape[1] ,3))
# all_segments[segm1] = (1, 0, 0)
# all_segments[segm2] = (0, 1, 0)
# all_segments[segm3] = (0, 0, 1) 
# all_segments[segm4] = (1, 1, 0)
# plt.figure()
# plt.imshow(all_segments)
# print(segm1)


# #clean segments
# segm1_opened = nd.binary_opening(segm1, np.ones((3,3)))
# segm1_closed = nd.binary_closing(segm1_opened, np.ones((3,3)))
# segm2_opened = nd.binary_opening(segm2, np.ones((3,3)))
# segm2_closed = nd.binary_closing(segm2_opened, np.ones((3,3)))
# segm3_opened = nd.binary_opening(segm3, np.ones((3,3)))
# segm3_closed = nd.binary_closing(segm3_opened, np.ones((3,3)))
# segm4_opened = nd.binary_opening(segm4, np.ones((3,3)))
# segm4_closed = nd.binary_closing(segm4_opened, np.ones((3,3)))
# all_segments_cleaned = np.zeros((image.shape[0], image.shape[1], 3))
# all_segments_cleaned[segm1_closed] = (1, 0, 0)
# all_segments_cleaned[segm2_closed] = (0, 1, 0)
# all_segments_cleaned[segm3_closed] = (0, 0, 1) 
# all_segments_cleaned[segm4_closed] = (1, 1, 0)
# plt.figure()
# plt.imshow(all_segments_cleaned)
# plt.show()


#define a function for histogram_based method
def hist_seg(x):
    #define a segmentation dictionary to store segmentation values
    segdic = {}
    #define a segmentation_cleaned dictionary to store smoothed sementation
    segclean = {}
    #initialise segmentations size
    all_segments = np.zeros((image.shape[0],image.shape[1] ,3))
    #initialise colors with combination of R,G,B
    colors = list(itertools.product([0, 1], repeat=3))
    for i in range(x):
        segdic['seg{}'.format(i+1)] = (image > (255/x*i)) & (image <= (255/x*i + 255/x))
        # segclean['segc_open{}'.format(i+1)] = nd.binary_opening(segdic['seg{}'.format(i+1)], np.ones((3,3)))
        # segclean['segc_close{}'.format(i+1)] = nd.binary_closing(segclean['segc_open{}'.format(i+1)], np.ones((3,3)))
        # all_segments[segclean['segc_close{}'.format(i+1)]] = colors[i+1]
        all_segments[segdic['seg{}'.format(i+1)]] = colors[i+1]
    
    plt.figure()
    plt.imshow(all_segments)
    plt.axis('off')
    plt.savefig("histogram.png",bbox_inches='tight',pad_inches=0.0)
    plt.show()


hist_seg(6)

print('done')