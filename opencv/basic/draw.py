import os
import cv2

img = cv2.imread("/home/anhoang/cv/opencv/drawing/wb.jpg")

print(img.shape)

#line
cv2.line(img, (100,150), (200,500), (0, 255, 0), 3)

#rectangle
cv2.rectangle(img, (200, 300), (400, 500), (0, 0, 255),5)

#circle
cv2.circle(img, (200, 300), 100, (0, 0, 255),5)


cv2.imshow('img', img)
cv2.waitKey(0)