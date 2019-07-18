import cv2
import numpy as np

path_to_picture = '../resources/the_book_thief.jpg'
img = cv2.imread(path_to_picture, cv2.IMREAD_GRAYSCALE)

# sift и surf запатентованы.
# При комерческом использвании этих алгоритмов надо платить каждый год.
sift = cv2.xfeatures2d.SIFT_create()
surf = cv2.xfeatures2d.SURF_create()  # более быстрый алгоритм

# orb бесплатный
orb = cv2.ORB_create(nfeatures=1500)

# kp = sift.detect(img, None) #найти keypoints
# keypoints, descriptors = sift.detectAndCompute(img, None)  # найти keypoints и descriptors
# keypoints, descriptors = surf.detectAndCompute(img, None)  # найти keypoints и descriptors
keypoints, descriptors = orb.detectAndCompute(img, None)  # найти keypoints и descriptors

# img = cv2.drawKeypoints(img, kp, None)
img = cv2.drawKeypoints(img, keypoints, None)

cv2.imshow('image', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
