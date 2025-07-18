import cv2 
import os, time
from HandTracking import handDetector

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)

cap.set(3, wCam)
cap.set(4, hCam)

folderPath = "finger_images"
myList = os.listdir(folderPath)
# print(myList)
overlayList = []

for img_path in myList:
    image = cv2.imread(f'{folderPath}/{img_path}')
    # print(f'{folderPath}/{img_path}')
    # image = cv2.resize(image, (200, 200))
    overlayList.append(image)

print(len(overlayList))

pTime = 0
cTime = 0

detector = handDetector(detectionCon=0.75)
tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if(len(lmList) != 0):
        fingers = []
        if(lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1]):
            fingers.append(1)
        else:
            fingers.append(0)
        for id in range(1, 5):
            if(lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]):
                fingers.append(1)
            else:
                fingers.append(0)
        totalFingers = fingers.count(1)
        print(totalFingers)

        h, w, c = overlayList[totalFingers - 1].shape
        img[0:h, 0:w] = overlayList[totalFingers - 1]
        # cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
        # cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN,  10, (255, 0, 0), 3)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img,f'FPS:{int(fps)}',(430,50),cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow('Image', img)
    cv2.waitKey(1)