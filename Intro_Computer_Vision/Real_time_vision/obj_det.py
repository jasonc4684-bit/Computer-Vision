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
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            width, height = x2-x1, y2-y1
            conf = int(box.conf[0] * 100)
            cls = box.cls[0]
            name = model.names[cls]
            cvzone.cornerRect(img, x1, y1, width, height)
            pos = (x1, y1-10)
            cvzone.putTextRect(img, f"{name} {conf}", pos, scale=1)

    if not success:
        break
    cv2.imshow("obj_det", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
