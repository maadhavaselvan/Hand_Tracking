import cv2
import mediapipe as mp
import time
import pyautogui
class HandTracking():
    def __init__(self,mode=False,maxHands=2,detectionCon=0.5,trackCon=0.5):
        self.mode=mode
        self.maxHands=maxHands
        self.detectionCon=detectionCon
        self.trackCon=trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode,max_num_hands=self.maxHands,min_detection_confidence=self.detectionCon,min_tracking_confidence=self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
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
                for id,lm in enumerate(myHand.landmark):
                    h,w,c=img.shape
                    cx,cy=int(lm.x*w),int(lm.y*h)
                    lmList.append([id,cx,cy])
        return lmList
def main():
    detector = HandTracking()
    cap=detector.camerasetting()
    pTime=0
    while True:
        success,img = cap.read()
        if success!=True:
            print("Could not read image Retrying...")
        img=detector.DrawHands(img)
        lmList=detector.findPosition(img)
        if len(lmList)>0:
            print(lmList)
            print("\n")
        CTime = time.time()
        fps = 1 / (CTime - pTime)
        pTime = CTime
        img = cv2.flip(img, 1)
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Image', img)
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break
    cv2.destroyAllWindows()
if __name__=="__main__":
    main()

