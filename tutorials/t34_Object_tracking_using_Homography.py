import cv2
import numpy as np

path_to_picture = "../resources/book_1001_zadacha.jpg"
img = cv2.imread(path_to_picture, cv2.IMREAD_GRAYSCALE)

# SIFT Features Detector
sift = cv2.xfeatures2d.SIFT_create()

# features of picture
kp_image, desc_image = sift.detectAndCompute(img, None)

# обозначить на picture точки kp_image (ValueError: too many values to unpack)
# img = cv2.drawKeypoints(img, kp_image, img)

cap = cv2.VideoCapture(0)

# FlannBasedMatcher - объект для матчинга с параметрами.
index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)

while True:
    _, frame = cap.read()

    # convert to gray
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # features of grayframe
    kp_grayframe, desc_grayframe = sift.detectAndCompute(grayframe, None)

    # обозначить на grayframe точки kp_grayframe (ValueError: too many values to unpack)
    # grayframe = cv2.drawKeypoints(grayframe, kp_grayframe, grayframe)

    # find matches using flann (FlannBasedMatcher - объект для матчинга)
    matches = flann.knnMatch(desc_image, desc_grayframe, k=2)

    # отбираем только хорошие совпадения
    good_points = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good_points.append(m)

    # создаём картинку, отображающую совпадения
    img_with_matching = cv2.drawMatches(img, kp_image, grayframe, kp_grayframe, good_points, grayframe)

    # Homography(Гомография) - перспективная трансформация
    # если число совпадений> 10, ищем гомографию, иначе - отображаем просто grayframe
    if len(good_points) > 10:

        #get coordinates of picture(query) and grayframe(train) keypoints
        query_pts = np.float32([kp_image[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
        train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)

        #находим гомографию, матрицу перспективной трансформации между двумя картинками(query и train)
        matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()

        # Perspective transform
        h, w = img.shape #размеры картинки(query)

        #создаём рамку для обозначения совпадающей картинки
        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)

        #делаем перспективную трансформацию рамки параллельно картинке
        dst = cv2.perspectiveTransform(pts, matrix)
        homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)

        cv2.imshow("Homography", homography)
    else:
        cv2.imshow("Homography", grayframe)

    cv2.imshow("Image", img)
    cv2.imshow("grayFrame", grayframe)
    cv2.imshow("img_with_matching", img_with_matching)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
