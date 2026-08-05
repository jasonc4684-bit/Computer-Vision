import cv2

Img = "Computer_vision/intro/school.jpg"
img = cv2.imread(Img)

#parameters for cv2.putText
point = (10,20)
cv2_Font = cv2.FONT_HERSHEY_DUPLEX
font_thickness = 1
textColor = (0,255,200)
display_thickness = 2

cv2.putText(img, "This is one high school at Virginia", 
            point, cv2_Font, thickness, textColor, display_thickness)

cv2.imshow("img", img)

cv2.waitKey(0)
