import numpy as np
import pandas as pd
import os, glob, csv
from tqdm import tqdm
import matplotlib.pyplot as plt
from detecta import detect_peaks


basedir = os.getcwd()
EMG_DIR = os.path.join(basedir, 'EMG_data')
EMG_FILE = sorted(glob.glob(f"{EMG_DIR}/*.csv"))
change_cols = ["102_S03_A", "102_S04_A", "102_S05_A", "102_S06_A", "102_S04_B", "102_S05_B", "102_S06_B","102_S01_B","102_S02_B"]
EMG = {}
ACC = {}

for file in tqdm(EMG_FILE):
    with open(file, mode='r',encoding='utf-8') as f:
        csv_reader = csv.reader(f)
        
        for idx, row in enumerate(csv_reader):
            if 'X [s]' in row:
                break
            
    df = pd.read_csv(file, header = idx)    

    EMG_COL = {}
    ACC_COL = {}

    if 'S09_A_109_resample' not in file:
        for c in df.columns:
            if 'EMG' in c:
                if f"{c.split('_')[2]}_{c.split('_')[3]}_{c.split('_')[4]}" not in EMG_COL.keys():
                    EMG_COL[f"{c.split('_')[2]}_{c.split('_')[3]}_{c.split('_')[4]}"] = []
                    
                EMG_COL[f"{c.split('_')[2]}_{c.split('_')[3]}_{c.split('_')[4]}"].append(c)
            
            elif 'Acc' in c:
                if f"{c.split('_')[2]}_{c.split('_')[3]}_{c.split('_')[4]}" not in ACC_COL.keys():
                    ACC_COL[f"{c.split('_')[2]}_{c.split('_')[3]}_{c.split('_')[4]}"] = []
                    
                ACC_COL[f"{c.split('_')[2]}_{c.split('_')[3]}_{c.split('_')[4]}"].append(c)
                
        for trial in EMG_COL:
                
            emg = df[EMG_COL[trial]]
            emg.columns = ['RF','VL','BF','GM']
            emg = emg.dropna()
            emg = emg - emg.mean()
            
            acc = df[ACC_COL[trial]]
            acc.columns = ['X [g]','Y [g]','Z [g]']
            acc = acc.dropna()
            acc['R [g]'] = np.sqrt(np.sum(acc**2,1))
            
            if trial in change_cols:
                trial = f"{trial.split('_')[1]}_{trial.split('_')[2]}_{trial.split('_')[0]}"
                
            EMG[trial] = emg
            ACC[trial] = acc

    else:
        emg = df[['Trigno IM sensor 2: EMG 2 (IM)->RESAMP [Volts]','Trigno IM sensor 3: EMG 3 (IM)->RESAMP [Volts]',
            'Trigno IM sensor 5: EMG 5 (IM)->RESAMP [Volts]','Trigno IM sensor 14: EMG 14 (IM)->RESAMP [Volts]']]
        emg.columns = ['RF','VL','BF','GM']
        emg = emg.dropna()
        emg = emg - emg.mean()
        
        acc = df[['Trigno IM sensor 14: Acc 14.X (IM)->RESAMP [g]',
                  'Trigno IM sensor 14: Acc 14.Y (IM)->RESAMP [g]',
                  'Trigno IM sensor 14: Acc 14.Z (IM)->RESAMP [g]']]
        acc.columns = ['X [g]','Y [g]','Z [g]']
        acc = acc.dropna()
        acc['R [g]'] = np.sqrt(np.sum(acc**2,1))
                               
        EMG['S09_A_109'] = emg
        ACC['S09_A_109'] = acc
        
for trial in tqdm(ACC):

    jump_idx = detect_peaks(ACC[trial]['R [g]'][:100*60], mph=8, mpd = 100*1, show=False, title=trial)[-1]
    jump_idx = jump_idx + (100 * 30) # 점프 후 30초 뒤
    EMG[trial] = EMG[trial][jump_idx * 10:].reset_index(drop=True)
    ACC[trial] = ACC[trial][jump_idx:].reset_index(drop=True)
    
    EMG[trial]['Time'] = np.arange(0, len(EMG[trial])/1000, 1/1000)[:len(EMG[trial])]
    ACC[trial]['Time'] = np.arange(0, len(ACC[trial])/100, 1/100)[:len(ACC[trial])]
    
    EMG[trial].to_csv(f"{basedir}/PROCESSING/EMG/{trial}.csv", index=False)
    ACC[trial].to_csv(f"{basedir}/PROCESSING/ACC/{trial}.csv", index=False)
    