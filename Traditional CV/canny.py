import cv2

#load denosied image
image = cv2.imread('image_denoised.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 600,600)
cv2.imshow('image', image)

#canny edge detection
edges = cv2.Canny(image, 30, 100)
cv2.namedWindow('edges', cv2.WINDOW_NORMAL)
cv2.resizeWindow('edges', 600,600)
cv2.imshow('edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

result=cv2.imwrite('canny.png', edges)


