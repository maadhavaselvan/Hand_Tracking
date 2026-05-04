import time
import numpy as np
import math
class OneEuro():
    def __init__(self,f_c_min=0.1,beta=0.007,f_c_min_vel=1):
        self.f_c_min = f_c_min  # min cutoff freq (Hz)
        self.beta = beta  # speed coefficient
        self.f_c_min_vel = f_c_min_vel  # derivative cutoff (Hz)
        self.c_prev    = None      # previous smoothed position
        self.smooth_prev_vel   =0        # previous smoothed velocity
        self.previousTime    = None          # previous timestamp
        #beta=0.007 if you use pixel dont forget
    def Time(self):
        currentTime = time.time()
        self.T=currentTime-self.previousTime
        if self.T==0:
            self.T=0.000001
        self.previousTime=currentTime
    def alpha(self,f):
        t=(2*math.pi*f*self.T)
        alpha=(t/(1+t))
        return alpha
    def filter(self,raw):
        c_raw=np.array(raw)
        if self.previousTime is None:
            self.c_prev=c_raw
            self.previousTime=time.time()
            return self.c_prev
        self.Time()
        raw_vel = (c_raw - self.c_prev) / self.T
        alpha_v  = self.alpha(self.f_c_min_vel)
        smooth_vel  = alpha_v * raw_vel + (1 - alpha_v) * self.smooth_prev_vel
        f_c = self.f_c_min + self.beta *np.linalg.norm(smooth_vel)
        alpha_c = self.alpha(f_c)
        smooth_c = alpha_c * c_raw + (1 - alpha_c) * self.c_prev
        self.c_prev  = smooth_c
        self.smooth_prev_vel = smooth_vel
        return smooth_c











