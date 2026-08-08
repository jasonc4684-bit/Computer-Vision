import cv2

video = "video.MP4"
vid = cv2.VideoCapture(video)

while True:
    # .read() returns two outputs
    success, img = vid.read()

    
    if not success:
        break

    cv2.imshow("Video of falling tree", img)
    cv2.waitKey(1)
    