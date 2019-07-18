import cv2
import numpy as np

path_to_picture1 = '../resources/the_book_thief.jpg'
img1 = cv2.imread(path_to_picture1, cv2.IMREAD_GRAYSCALE)

path_to_picture2 = '../resources/tutor_holding_book.jpg'
img2 = cv2.imread(path_to_picture2, cv2.IMREAD_GRAYSCALE)

# sift и surf запатентованы.

# orb Detector бесплатный
# orb = cv2.ORB_create(nfeatures=1500)
orb = cv2.ORB_create()

keypoints1, descriptors1 = orb.detectAndCompute(img1, None)  # найти keypoints и descriptors
keypoints2, descriptors2 = orb.detectAndCompute(img2, None)  # найти keypoints и descriptors

# мы сравниваем дескрипторы независимо от масштаба, угла поворота, освещённости.

for d in descriptors1:
    print(d)

# Brute Force Matching
# вместе с orb детектором обычно используется cv2.NORM_HAMMING
# crossCheck=True - означает, что будет меньше совпадений, но более качественных
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)
matches = sorted(matches, key=lambda x: x.distance)  # sort matches with distance

print(len(matches))

for m in matches:
    print(m.distance)

# из массива matches берём первые 20 совпадений
# flags=2 - означает, что показываем только совпадающие features
matching_result = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:20], None, flags=2)

# img = cv2.drawKeypoints(img, kp, None)
# img = cv2.drawKeypoints(img, keypoints, None)

cv2.imshow('image1', img1)
cv2.imshow('image2', img2)
cv2.imshow('Matching result', matching_result)

cv2.waitKey(0)
cv2.destroyAllWindows()
