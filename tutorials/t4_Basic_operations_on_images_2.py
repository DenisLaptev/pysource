import cv2
import numpy as np

path_to_image = '../resources/flag.png'
image = cv2.imread(path_to_image)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

rows, cols, chs = image.shape
roi = image[100:rows, 0:400]

gray_image[175, 100] = 255

print(image)
print(image.shape)
print(image[175, 300])

image[175, 300] = (255, 0, 0)
image[175, 301] = (255, 0, 0)
image[175, 302] = (255, 0, 0)
image[175, 303] = (255, 0, 0)
image[175, 304] = (255, 0, 0)

cv2.imshow("image", image)
cv2.imshow("gray_image", gray_image)
cv2.imshow("roi", roi)

cv2.waitKey(0)
cv2.destroyAllWindows()
