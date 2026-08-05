import cv2

vid = cv2.VideoCapture("video.MP4")

while True:
    success, img = vid.read()

    if not success:
        break
    cv2.imshow("Video of falling tree", img)
    cv2.waitKey(1)
    