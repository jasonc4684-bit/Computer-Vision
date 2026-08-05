import cv2

vid = cv2.VideoCapture(0)

while True:
    success, img = vid.read()

    if not success:
        break
    cv2.imshow("Video of falling tree", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    