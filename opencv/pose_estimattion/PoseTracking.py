import cv2 
import mediapipe as mp
import time
import math

class PoseDetector():
    def __init__(self, mode=False, upBody=False, smooth=True,
                 detectionCon=0.5, trackingCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
                        static_image_mode=self.mode,
                        model_complexity=1,
                        smooth_landmarks=self.smooth,
                        enable_segmentation=False,
                        smooth_segmentation=True,
                        min_detection_confidence=self.detectionCon,
                        min_tracking_confidence=self.trackingCon
                        )
        
    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS,
                                           self.mpDraw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                                           self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                                           )

        return img

    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList
    
    def findAngle(self, img, p1, p2, p3, draw=True):
        # Get the landmark
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]
        # Calculate the angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360
        # Draw
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x2, y2), (x3, y3), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        return angle

def main():
    cap = cv2.VideoCapture('PoseVideo/1.mp4')
    detector = PoseDetector()
    ptime = 0
    while True:
        success, img = cap.read()
        img =detector.findPose(img, draw=False)
        lmList = detector.findPosition(img, draw=False)
        if len(lmList) != 0:
            # print(lmList[14])
            # cv2.circle(img, (lmList[14][1], lmList[14][2]), 20, (0, 0, 255), cv2.FILLED)
            angle = detector.findAngle(img, 11, 13, 15)
            print(angle)
        ctime = time.time()
        fps = 1/(ctime - ptime)
        ptime =  ctime

        cv2.putText(img, f'FPS: {int(fps)}', (350, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow('img', img)
        cv2.waitKey(1)

if __name__ == '__main__':
    main()
