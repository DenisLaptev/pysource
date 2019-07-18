import cv2
import numpy as np

# from WebCam
# cap = cv2.VideoCapture(0)

# from mp4-file
path_to_video = '../resources/red_panda_snow.mp4'
cap = cv2.VideoCapture(path_to_video)

# coder to save video
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("outputvideo.avi", fourcc, 25, (640, 360))

while True:
    ret, frame = cap.read()
    # print size of the frame
    print(frame.shape)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # mirror reflection
    frame2 = cv2.flip(frame, 0)
    frame3 = cv2.flip(frame, 1)

    cv2.imshow('frame', frame)
    cv2.imshow('gray_frame', gray_frame)
    cv2.imshow('frame2', frame2)
    cv2.imshow('frame3', frame3)

    out.write(frame3)

    # wait for any key 1ms and goto while True
    # cv2.waitKey(0) - don`t wait, just press any key
    key = cv2.waitKey(25)
    if key == 27:
        break

out.release()
cap.release()
cv2.destroyAllWindows()
