import cv2
import mediapipe as mp
import time
import math
import numpy as np
import pyautogui
import OneEuro
pyautogui.PAUSE = 0
pyautogui.FAILSAFE=False
class HandTracking():
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
        self.prevtime=time.time()
        self.euro_filter = OneEuro.OneEuro()
        self.frameend=100
    def camerasetting(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(3,640)
        self.cap.set(4,480)
        if self.cap.isOpened()!=True:
            print("Could not open camera,Please check if camera is connected")
        return self.cap
    def DrawHands(self,img,draw=False):
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
        screen_x = np.interp(lmList[8][0]*640,[self.frameend,640-self.frameend],[0,HandTracking.width-1])
        screen_y = np.interp(lmList[8][1]*480, [self.frameend,480-self.frameend],[0,HandTracking.height-1])
        smooth = self.euro_filter.filter([screen_x, screen_y])
        pyautogui.moveTo(int(smooth[0]),int(smooth[1]))
        distindexthumb=math.dist(lmList[5],lmList[4])
        distpalm=math.dist(lmList[9],lmList[0])
        ratio=distindexthumb/distpalm
        if(ratio<0.35 and ((time.time()-self.prevtime)>0.3)):
            time.sleep(0.05)
            pyautogui.click()
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
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break
        img=detector.DrawHands(img)
        lmList=detector.findPosition(img)
        if len(lmList)>9:
            if(math.dist(lmList[8], lmList[12])/(math.dist(lmList[9],lmList[0]))>0.33):
                detector.mousepointer(lmList)
    cap.release()
    cv2.destroyAllWindows()
if __name__=="__main__":
    main()

