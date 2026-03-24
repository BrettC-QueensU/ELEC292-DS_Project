import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Get all CSV files from Brett Jumping Folder
files = glob.glob('Raw_Data/Brett_RawData/*.csv')

# Preprocessing Brett Data
for filepath in files:
    #import csv file as data frame, and interpolate to fill any missing values
    df = pd.read_csv(filepath)
    df = df.interpolate(method = 'linear', inPlace = True)
    df_acc = df.drop(columns = 'Time (s)')

    #use a rolling mean on the data, with a
    window_size = 21
    y_sma = df_acc.rolling(window_size).mean()

    #replacing the acceleration columns in the data frame with the ones in y_sma
    df['Linear Acceleration x (m/s^2)'] = y_sma['Linear Acceleration x (m/s^2)']
    df['Linear Acceleration y (m/s^2)'] = y_sma['Linear Acceleration y (m/s^2)']
    df['Linear Acceleration z (m/s^2)'] = y_sma['Linear Acceleration z (m/s^2)']
    df['Absolute acceleration (m/s^2)'] = y_sma['Absolute acceleration (m/s^2)']

    #delete rows with na at the start resulting from applying sma
    df_processed = df.dropna()

    #creating a new csv file with the pre-processed data
    filename = filepath.split('_RawData/')[1].split("_RawData.csv")[0]
    df_processed.to_csv('Pre-Processed_Data/Brett/' + filename + '_PreProcessed.csv', index = False)

#Visualisation used to aid with trial and error of window size
#time = df['Time (s)']
#ax = y_sma['Linear Acceleration x (m/s^2)']
#ay = y_sma['Linear Acceleration y (m/s^2)']
#az = y_sma['Linear Acceleration z (m/s^2)']
#aa = y_sma['Absolute acceleration (m/s^2)']

#fig, jump = plt.subplots(figsize=(12, 5))

#jump.plot(time, ax, label='X', color='tab:blue', linewidth=1.2)
#jump.plot(time, ay, label='Y', color='tab:orange', linewidth=1.2)
#jump.plot(time, az, label='Z', color='tab:green', linewidth=1.2)
#jump.plot(time, aa, label='Absolute', color='tab:purple', linewidth=1.2)
#jump.set_xlabel('Time (s)', fontsize=12)
#jump.set_ylabel('Linear Acceleration (m/s²)', fontsize=12)
#jump.set_title('Linear Acceleration vs Time (X, Y, Z, Absolute)', fontsize=14)
#jump.legend(title='Axis', fontsize=12)
#jump.grid(True, linestyle='--', alpha=0.5)
#jump.axhline(0, color='black', linewidth=0.8, linestyle='-')

#plt.tight_layout()
#plt.show()


