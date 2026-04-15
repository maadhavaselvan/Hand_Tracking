import cv2
import mediapipe as mp
import time
import math
import numpy as np
import pyautogui
pyautogui.PAUSE = 0
class HandTracking():
    slow_mode_counter = 0
    width,height=pyautogui.size()
    smooth_factor=0.3
    def __init__(self,mode=False,maxHands=1,detectionCon=0.5,trackCon=0.5):
        self.mode=mode
        self.maxHands=maxHands
        self.detectionCon=detectionCon
        self.trackCon=trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode,max_num_hands=self.maxHands,min_detection_confidence=self.detectionCon,min_tracking_confidence=self.trackCon,model_complexity=0)
        self.mpDraw = mp.solutions.drawing_utils
        self.screen_x_old, self.screen_y_old = 640, 480
        self.prevtime=time.time()
    def camerasetting(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3,640)
        self.cap.set(4,480)
        if self.cap.isOpened()!=True:
            print("Could not open camera,Please check if camera is connected")
        return self.cap
    def DrawHands(self,img,draw=True):
        RGBimg=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results=self.hands.process(RGBimg)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img,handLms,self.mpHands.HAND_CONNECTIONS)
        return img
    def findPosition(self,img,draw=True):
        lmList=[]
        if self.results.multi_hand_landmarks:
            for myHand in self.results.multi_hand_landmarks:
                for lm in myHand.landmark:
                    lmList.append([lm.x,lm.y,lm.z])
        return lmList
    def mousepointer(self,lmList):
        screen_x = np.interp(lmList[8][0]*640,[0,640],[0,HandTracking.width])
        screen_y = np.interp(lmList[8][1]*480, [0,480],[0,HandTracking.height])
        screen_x=(screen_x*HandTracking.smooth_factor)+(self.screen_x_old*(1-HandTracking.smooth_factor))
        screen_y=(screen_y*HandTracking.smooth_factor)+(self.screen_y_old*(1-HandTracking.smooth_factor))
        self.screen_y_old=screen_y
        self.screen_x_old=screen_x
        pyautogui.moveTo(int(screen_x),int(screen_y))
        distindexthumb=math.dist(lmList[8],lmList[4])
        distpalm=math.dist(lmList[9],lmList[0])
        ratio=distindexthumb/distpalm
        if(ratio<0.25 and ((time.time()-self.prevtime)>0.3)):
            pyautogui.click(int(screen_x),int(screen_y))
            self.prevtime=time.time()
            print("Click")
def main():
    detector = HandTracking()
    cap=detector.camerasetting()
    pTime=0
    while True:
        success,img = cap.read()
        if success!=True:
            print("Could not read image Retrying...")
        img = cv2.flip(img, 1)
        img=detector.DrawHands(img)
        lmList=detector.findPosition(img)
        if len(lmList)>9:
            detector.mousepointer(lmList)
        CTime = time.time()
        fps = 1 / (CTime - pTime)
        pTime = CTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Image', img)

        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break
    cv2.destroyAllWindows()
if __name__=="__main__":
    main()

