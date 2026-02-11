import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

fs = 1000
base_path = os.getcwd()

emg_path = os.path.join(base_path,"DATA","EMG")
imu_path = os.path.join(base_path,"DATA","IMU")
save_base = os.path.join(base_path,"SEGMENTED")

folders = ["STANCE/LEFT","STANCE/RIGHT","SWING/LEFT","SWING/RIGHT"]
for f in folders:
    os.makedirs(os.path.join(save_base,f), exist_ok=True)

files = os.listdir(emg_path)

for file in files:
    
    emg = pd.read_excel(os.path.join(emg_path,file))
    imu = pd.read_excel(os.path.join(imu_path,file))
    
    emg_left  = emg.iloc[:,1:5]
    emg_right = emg.iloc[:,5:9]
    
    knee_L = imu.iloc[:,1].values
    knee_R = imu.iloc[:,2].values
    
    knee_L = savgol_filter(knee_L,51,3)
    knee_R = savgol_filter(knee_R,51,3)
    
    vel_L = np.gradient(knee_L)*fs
    vel_R = np.gradient(knee_R)*fs
    
    vel_L = savgol_filter(vel_L,31,3)
    vel_R = savgol_filter(vel_R,31,3)
    
    zc_L = np.where(np.diff(np.sign(vel_L))!=0)[0]
    zc_R = np.where(np.diff(np.sign(vel_R))!=0)[0]
    
    HS_L, TO_L = [], []
    HS_R, TO_R = [], []
    
    for i in range(1,len(zc_L)-1):
        idx = zc_L[i]
        if vel_L[idx-1]>0 and vel_L[idx+1]<0:
            HS_L.append(idx)
        elif vel_L[idx-1]<0 and vel_L[idx+1]>0:
            TO_L.append(idx)
    
    for i in range(1,len(zc_R)-1):
        idx = zc_R[i]
        if vel_R[idx-1]>0 and vel_R[idx+1]<0:
            HS_R.append(idx)
        elif vel_R[idx-1]<0 and vel_R[idx+1]>0:
            TO_R.append(idx)
    
    stance_L = []
    swing_L  = []
    stance_R = []
    swing_R  = []
    
    for i in range(min(len(HS_L)-1,len(TO_L))):
        if HS_L[i]<TO_L[i]:
            stance_L.append(emg_left.iloc[HS_L[i]:TO_L[i]])
        if i+1<len(HS_L) and TO_L[i]<HS_L[i+1]:
            swing_L.append(emg_left.iloc[TO_L[i]:HS_L[i+1]])
    
    for i in range(min(len(HS_R)-1,len(TO_R))):
        if HS_R[i]<TO_R[i]:
            stance_R.append(emg_right.iloc[HS_R[i]:TO_R[i]])
        if i+1<len(HS_R) and TO_R[i]<HS_R[i+1]:
            swing_R.append(emg_right.iloc[TO_R[i]:HS_R[i+1]])
    
    stance_L = pd.concat(stance_L) if stance_L else pd.DataFrame()
    swing_L  = pd.concat(swing_L) if swing_L else pd.DataFrame()
    stance_R = pd.concat(stance_R) if stance_R else pd.DataFrame()
    swing_R  = pd.concat(swing_R) if swing_R else pd.DataFrame()
    
    stance_L.to_excel(os.path.join(save_base,"STANCE/LEFT",file),index=False)
    swing_L.to_excel(os.path.join(save_base,"SWING/LEFT",file),index=False)
    stance_R.to_excel(os.path.join(save_base,"STANCE/RIGHT",file),index=False)
    swing_R.to_excel(os.path.join(save_base,"SWING/RIGHT",file),index=False)

print("Segmentation Completed")
