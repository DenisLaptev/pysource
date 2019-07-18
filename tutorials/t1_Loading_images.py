import cv2

path_to_picture = '../resources/book_1001_zadacha.jpg'
path_to_outputfile = '../resources/output.jpg'

image = cv2.imread(path_to_picture)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow('picture', image)
cv2.imshow('gray picture', gray_image)

cv2.imwrite(path_to_outputfile, gray_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
