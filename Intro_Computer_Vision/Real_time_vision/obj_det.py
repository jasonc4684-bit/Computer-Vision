from ultralytics import YOLO
import cv2
import cvzone

#capture local web cam, adjust webcam if needed
cam = 0
vid = cv2.VideoCapture(cam)

yoloModel = "yolo26m.pt" # change "s" if larger data is needed (ex."m")
model = YOLO(yoloModel)

while True:
    success, img = vid.read()
    results = model(img, stream=True)
    for result in results:
        for box in result.bxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

    if not success:
        break
    cv2.imshow("obj_det", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
