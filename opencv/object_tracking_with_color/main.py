import cv2
from util import get_limits
from PIL import Image

red = [0, 255, 255] #red in brg

cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()

    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    lower, upper = get_limits(red)

    mask = cv2.inRange(hsvImage,lower, upper)

    mask_ = Image.fromarray(mask)
    
    bbox = mask_.getbbox() #Coordinate of bounding box
    
    if bbox is not None:
        x1, y1, x2, y2 = bbox

        frame = cv2.rectangle(frame, (x1,y1), (x2,y2), (0, 255, 0), 5) #Draw green rectangle bounding box

    print(bbox)

    cv2.imshow("frame", frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()