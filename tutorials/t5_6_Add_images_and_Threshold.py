import cv2
import numpy as np

# to add images they have to be the same size
path_to_img1 = '../resources/road.jpg'
path_to_img2 = '../resources/car.jpg'

image1 = cv2.imread(path_to_img1)
image2 = cv2.imread(path_to_img2)

gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# to add images they have to be the same size
print('image1.shape=', image1.shape)
print('image2.shape=', image2.shape)

sum = cv2.add(image1, image2)
plus = image1 + image2
weighted_sum = cv2.addWeighted(image1, 1, image2, 0.5, 0)
ret, threshold = cv2.threshold(gray_image2, 250, 255, cv2.THRESH_BINARY)
threshold_inv = cv2.bitwise_not(threshold)

# вырезаем машину из image2(убираем всё остальное)
car_just = cv2.add(image2, image2, mask=threshold_inv)

road_background = cv2.bitwise_and(image1, image1, mask=threshold)

result = cv2.add(car_just,road_background)

# cv2.imshow('road', image1)
# cv2.imshow('car', image2)
# cv2.imshow('gray_image2', gray_image2)
# cv2.imshow('threshold', threshold)
# cv2.imshow('sum', sum)
# cv2.imshow('plus', plus)
# cv2.imshow('weighted_sum', weighted_sum)
cv2.imshow('sum_with_mask', car_just)
cv2.imshow('road', road_background)
cv2.imshow('result', result)

cv2.waitKey(0)
cv2.destroyAllWindows()
