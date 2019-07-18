import cv2
import numpy as np

path_to_image = '../resources/book_1001_zadacha.jpg'
image = cv2.imread(path_to_image)

print(image.shape)

cv2.line(image, (10, 10), (100, 100), (255, 255, 0), 5)
cv2.circle(image, (80, 170), 10, (0, 255, 0), 6)
cv2.rectangle(image, (150, 150), (200, 200), (0, 0, 255), 2)
cv2.ellipse(image, (80, 85), (80, 20), 30, 0, 300, (200, 200, 20), 3)
points = np.array([[[120, 130], [200, 130], [32, 25], [28, 25]]], np.int32)
cv2.polylines(image, [points], True, (0, 200, 200), 8)

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(image, "book", (20, 100), font, 1, (0, 0, 250), 3)

cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
